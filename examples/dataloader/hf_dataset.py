# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# type: ignore
"""HuggingFace Dataset implementation for language model training."""
# Standard Library
import os
from pathlib import Path
from typing import List, Union

# Third Party
import torch
from datasets import interleave_datasets, load_dataset, load_from_disk
from torch.utils.data.dataloader import default_collate

# First Party
from hyperpod_checkpointless_training.inprocess.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


class DataTypes:
    """Data format constants."""

    ARROW = ".arrow"
    JSONGZ = ".json.gz"
    JSON = ".json"


class HuggingFaceDataset:
    """
    A Dataset class that loads HuggingFace tokenized datasets from disk.

    This dataset is designed to work with pre-tokenized language model datasets
    stored in HuggingFace format on disk, supporting multiple data formats.
    """

    def __init__(
        self,
        input_path: str,
        partition: str = "train",
        seq_length: int = 8192,
        keep_in_memory: bool = True,
    ):
        """
        Initialize the HF Tokenized Dataset.

        Args:
            input_path: Path(s) to the HuggingFace dataset directory
            partition: Dataset partition to load (e.g., "train", "validation")
            seq_length: Sequence length
        """
        self.input_path = input_path
        self.partition = partition
        self.seq_length = seq_length
        self.data_format = self._get_data_format(self.input_path)
        self._dataset = None

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.long)

        logger.info(
            f"Loading HuggingFace dataset from {input_path} with format {self.data_format}"
        )

        self._dataset = self.fetch_dataset(self.input_path, keep_in_memory)
        logger.info(f"Loaded dataset with {len(self._dataset)} samples")

    def fetch_dataset(self, path, keep_in_memory):
        """
        Fetch dataset from the given path based on data format.

        Args:
            path: Path to the dataset

        Returns:
            Loaded dataset
        """
        match self.data_format:
            case DataTypes.ARROW:
                dataset = load_from_disk(path, keep_in_memory=keep_in_memory)
            case DataTypes.JSONGZ:
                dataset = load_dataset(
                    "json",
                    data_files=[os.path.join(path, f"*{DataTypes.JSONGZ}")],
                    split=self.partition,
                )
            case DataTypes.JSON:
                dataset = load_dataset(
                    "json",
                    data_files=[os.path.join(path, f"*{DataTypes.JSON}")],
                    split=self.partition,
                )
            case _:
                raise NotImplementedError(f"{self.data_format} is not supported.")
        return dataset

    def _get_data_format(self, path):
        """
        Determine the data format based on file extensions in the path.

        Args:
            path: Path(s) to check

        Returns:
            Data format type
        """
        if isinstance(path, str):
            path = [path]
        files = []
        for p in path:
            if os.path.exists(p):
                files += [f for f in Path(p).iterdir() if f.is_file()]

        if not files:
            # If no files found, assume ARROW format (for load_from_disk)
            return DataTypes.ARROW

        suffixes_list = list(set(["".join(Path(f).suffixes) for f in files]))

        if any(suffix == DataTypes.ARROW for suffix in suffixes_list):
            return DataTypes.ARROW
        elif any(suffix == DataTypes.JSONGZ for suffix in suffixes_list):
            return DataTypes.JSONGZ
        elif any(suffix == DataTypes.JSON for suffix in suffixes_list):
            return DataTypes.JSON
        else:
            raise NotImplementedError(
                f"Unsupported file format in dataset directory. Expecting files of type '.arrow' '.json.gz' or '.json' but instead found {','.join(suffixes_list)}."
            )

    @property
    def dataset(self):
        """Get the underlying dataset."""
        return self._dataset

    def __getitem__(self, index: int) -> dict:
        """
        Get item from dataset with indexing for tokens and labels.

        Args:
            index: Index of the item to retrieve

        Returns:
            Dictionary containing tokens, labels, loss_mask, and position_ids
        """
        obj = self._dataset[index]
        input_ids = torch.tensor(obj["input_ids"], dtype=torch.long)[: self.seq_length]

        labels = torch.tensor(obj.get("labels", obj["input_ids"]), dtype=torch.long)[
            : self.seq_length
        ]

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self._dataset)

    def collate_fn(self, batch):
        """
        Collate function for DataLoader to batch samples.

        Args:
            batch: List of samples to collate

        Returns:
            Collated batch using default_collate
        """
        return default_collate(batch)
