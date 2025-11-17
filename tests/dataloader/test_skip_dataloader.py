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
