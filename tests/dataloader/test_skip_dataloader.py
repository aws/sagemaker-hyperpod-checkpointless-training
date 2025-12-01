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
import unittest
from torch.utils.data import Dataset


class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return f"batch_{idx}"


class TestSkipDataLoader(unittest.TestCase):

    def test_init_default_skip_batches(self):
        from hyperpod_checkpointless_training.dataloader.skip_dataloader import SkipDataLoader

        dataset = MockDataset(5)
        loader = SkipDataLoader(dataset)

        self.assertEqual(loader._skip_batches, 0)

    def test_init_with_args_and_kwargs(self):
        from hyperpod_checkpointless_training.dataloader.skip_dataloader import SkipDataLoader

        dataset = MockDataset(5)
        loader = SkipDataLoader(dataset, batch_size=2, _skip_batches=1, shuffle=False)

        self.assertEqual(loader._skip_batches, 1)

    def test_iter_no_skip(self):
        from hyperpod_checkpointless_training.dataloader.skip_dataloader import SkipDataLoader

        dataset = MockDataset(3)
        loader = SkipDataLoader(dataset, batch_size=1, _skip_batches=0)

        batches = list(loader)
        self.assertEqual(len(batches), 3)

    def test_iter_skip_exact_number_of_batches(self):
        from hyperpod_checkpointless_training.dataloader.skip_dataloader import SkipDataLoader

        dataset = MockDataset(3)
        loader = SkipDataLoader(dataset, batch_size=1, _skip_batches=3)

        batches = list(loader)
        self.assertEqual(len(batches), 0)
