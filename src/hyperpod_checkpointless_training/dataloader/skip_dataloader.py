# type: ignore
from torch.utils.data import DataLoader


class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        _skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, *args, _skip_batches=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_batches = _skip_batches

    @property
    def skip_batches(self):
        return self._skip_batches

    @skip_batches.setter
    def skip_batches(self, num_batches):
        self._skip_batches = num_batches

    def __iter__(self):
        batch_count = 0
        for batch in super().__iter__():
            batch_count += 1

            # Skip the first skip_batches batches
            if batch_count <= self._skip_batches:
                continue
            yield batch
