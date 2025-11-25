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
import os
import shutil
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyperpod_checkpointless_training.dataloader.mmap.cache import (
    MMAPCache,
    TTLMMAPCache,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import RestartMode


class TestMMAPCache(unittest.TestCase):

    def test_init_method(self):
        cache = MMAPCache(
            cache_dir="/test/cache",
            batch_prefix="batch",
            lookback_length=5,
            prefetch_length=3,
            model_checkpoint_frequency=100,
            read_only=True,
            force_cold_start=False,
            pdl_per_node_group=None,
        )

        self.assertEqual(cache.cache_dir, "/test/cache")
        self.assertEqual(cache.batch_prefix, "batch")
        self.assertEqual(cache.max_cache_size, 8)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.dirname")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.basename")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.normpath")
    def test_validate_cache_dir_method(
        self, mock_normpath, mock_basename, mock_dirname, mock_listdir, mock_exists
    ):
        cache = MMAPCache(
            cache_dir="/test/cache/train", read_only=False, pdl_per_node_group=None
        )

        mock_normpath.return_value = "/test/cache/train"
        mock_basename.return_value = "train"
        mock_dirname.return_value = "/test/cache"
        mock_listdir.return_value = ["train", "other"]

        cache._validate_cache_dir()

        mock_normpath.assert_called_once_with("/test/cache/train")
        mock_basename.assert_called_once_with("/test/cache/train")
        mock_dirname.assert_called_once_with("/test/cache/train")
        mock_listdir.assert_called_once_with("/test/cache")

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.dirname")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.basename")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.normpath")
    @patch("torch.distributed.all_reduce")
    def test_validate_cache_dir_method_pdl_per_node_group(
        self,
        mock_all_reduce,
        mock_normpath,
        mock_basename,
        mock_dirname,
        mock_listdir,
        mock_exists,
    ):
        cache = MMAPCache(
            cache_dir="/test/cache/train", read_only=False, pdl_per_node_group=Mock()
        )

        mock_normpath.return_value = "/test/cache/train"
        mock_basename.return_value = "train"
        mock_dirname.return_value = "/test/cache"
        mock_listdir.return_value = ["train", "other"]
        cache._validate_cache_dir()

        mock_normpath.assert_called_once_with("/test/cache/train")
        mock_basename.assert_called_once_with("/test/cache/train")
        mock_dirname.assert_called_once_with("/test/cache/train")
        mock_listdir.assert_called_once_with("/test/cache")

        mock_all_reduce.assert_called_once()

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.dirname")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.basename")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.normpath")
    @patch("torch.distributed.all_reduce")
    def test_validate_cache_dir_method_exception_filenotfound(
        self,
        mock_all_reduce,
        mock_normpath,
        mock_basename,
        mock_dirname,
        mock_listdir,
        mock_exists,
    ):
        cache = MMAPCache(
            cache_dir="/test/cache/train", read_only=False, pdl_per_node_group=Mock()
        )

        mock_normpath.return_value = "/test/cache/train"
        mock_basename.return_value = "train"
        mock_dirname.return_value = "/test/cache"
        mock_listdir.return_value = ["train", "other"]
        mock_listdir.side_effect = FileNotFoundError()
        cache._validate_cache_dir()

        mock_dirname.assert_called_once_with("/test/cache/train")
        mock_listdir.assert_called_once_with("/test/cache")

        mock_all_reduce.assert_not_called()

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.makedirs")
    def test_init_read_only_true(self, mock_makedirs):
        cache = MMAPCache(cache_dir="/test/cache", read_only=True)
        cache.init()
        mock_makedirs.assert_not_called()

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.makedirs")
    @patch.object(MMAPCache, "_validate_cache_dir")
    @patch.object(MMAPCache, "_try_removing_cache_dir")
    @patch.object(MMAPCache, "_delete_dir")
    def test_init_read_only_false(
        self, mock_delete_dir, mock_try_removing, mock_validate, mock_makedirs
    ):
        cache = MMAPCache(cache_dir="/test/cache", read_only=False)
        cache.init()

        mock_validate.assert_called_once()
        mock_try_removing.assert_called_once()
        self.assertEqual(mock_delete_dir.call_count, 2)
        self.assertEqual(mock_makedirs.call_count, 3)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_len_method(self, mock_listdir):
        mock_listdir.return_value = [
            "batch_1.pt",
            "batch_2.pt",
            "batch_5.pt",
            "other_file.txt",
        ]
        cache = MMAPCache(cache_dir="/test/cache")

        result = len(cache)
        self.assertEqual(result, 3)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_get_content_indices(self, mock_listdir):
        mock_listdir.return_value = [
            "batch_1.pt",
            "batch_2.pt",
            "batch_5.pt",
            "other_file.txt",
        ]
        cache = MMAPCache(cache_dir="/test/cache")

        result = cache.get_content_indices()
        self.assertEqual(sorted(result), [1, 2, 5])

    def test_get_batch_filename(self):
        cache = MMAPCache(cache_dir="/test/cache", batch_prefix="test_batch")
        result = cache._get_batch_filename(42)
        self.assertEqual(result, "/test/cache/test_batch_42.pt")

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.tempfile.NamedTemporaryFile"
    )
    def test_create_staging_entry(self, mock_tempfile):
        mock_file = MagicMock()
        mock_file.name = "/test/cache/tmp/tmpfile123"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        cache = MMAPCache(cache_dir="/test/cache", read_only=False)
        result = cache.create_staging_entry()

        self.assertEqual(result, "/test/cache/tmp/tmpfile123")
        mock_tempfile.assert_called_once_with(dir="/test/cache/tmp", delete=False)

    @patch("torch.save")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.tempfile.NamedTemporaryFile"
    )
    def test_stage_content(self, mock_tempfile, mock_torch_save):
        mock_file = MagicMock()
        mock_file.name = "/test/cache/tmp/tmpfile123"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        cache = MMAPCache(cache_dir="/test/cache", read_only=False)
        content = "dummy content"
        staging_path = cache.stage_content(content)
        self.assertTrue(staging_path.startswith("/test/cache/tmp/"))
        mock_torch_save.assert_called_once_with(content, staging_path)

    def test_create_staging_entry_read_only(self):
        cache = MMAPCache(cache_dir="/test/cache", read_only=True)

        with self.assertRaises(ValueError):
            cache.create_staging_entry()

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.rename")
    def test_promote_content(self, mock_rename):
        cache = MMAPCache(cache_dir="/test/cache", read_only=False)
        cache.promote_content("/tmp/staging_file", 42)

        mock_rename.assert_called_once_with(
            "/tmp/staging_file", "/test/cache/batch_42.pt"
        )

    def test_promote_content_read_only(self):
        cache = MMAPCache(cache_dir="/test/cache", read_only=True)

        with self.assertRaises(ValueError):
            cache.promote_content("/tmp/staging_file", 42)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.torch.load")
    def test_get_content(self, mock_torch_load):
        mock_tensor = {"data": torch.tensor([1, 2, 3])}
        mock_torch_load.return_value = mock_tensor

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache.get_content(42)

        self.assertEqual(result, mock_tensor)
        mock_torch_load.assert_called_once_with("/test/cache/batch_42.pt", mmap=True)

    @patch.object(MMAPCache, "get_content_indices")
    @patch.object(MMAPCache, "_delete_file")
    def test_remove_oldest_no_removal_needed(self, mock_delete_file, mock_get_indices):
        mock_get_indices.return_value = [1, 2, 3]
        cache = MMAPCache(cache_dir="/test/cache", lookback_length=5, prefetch_length=5)

        cache._remove_oldest(10)
        mock_delete_file.assert_not_called()

    @patch.object(MMAPCache, "get_content_indices")
    @patch.object(MMAPCache, "_delete_file")
    def test_remove_oldest_removes_file(self, mock_delete_file, mock_get_indices):
        mock_get_indices.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        cache = MMAPCache(cache_dir="/test/cache", lookback_length=5, prefetch_length=5)

        cache._remove_oldest(10)
        mock_delete_file.assert_called_once_with("/test/cache/batch_1.pt")

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.remove")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_file_success(self, mock_filelock, mock_remove, mock_exists):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_file("/test/file.pt")

        self.assertTrue(result)
        mock_remove.assert_called_once_with("/test/file.pt")

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_file_not_exists(self, mock_filelock, mock_exists):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = False

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_file("/test/file.pt")

        self.assertTrue(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    @patch("os.remove")
    def test_delete_file_exception_filenotfound(
        self, mock_remove, mock_filelock, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_remove.side_effect = FileNotFoundError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_file("/test/file.pt")

        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    @patch("os.remove")
    def test_delete_file_exception_isadirectory(
        self, mock_remove, mock_filelock, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_remove.side_effect = IsADirectoryError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_file("/test/file.pt")

        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    @patch("os.remove")
    def test_delete_file_exception_oserror(
        self, mock_remove, mock_filelock, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_remove.side_effect = OSError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_file("/test/file.pt")

        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.shutil.rmtree")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_dir_success(self, mock_filelock, mock_rmtree, mock_exists):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_dir("/test/dir")

        mock_rmtree.assert_called_once_with("/test/dir")
        self.assertTrue(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.shutil.rmtree")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_dir_exception_filenotfound(
        self, mock_filelock, mock_rmtree, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_rmtree.side_effect = FileNotFoundError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_dir("/test/dir")

        mock_rmtree.assert_called_once_with("/test/dir")
        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.shutil.rmtree")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_dir_exception_notadirectory(
        self, mock_filelock, mock_rmtree, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_rmtree.side_effect = NotADirectoryError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_dir("/test/dir")

        mock_rmtree.assert_called_once_with("/test/dir")
        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.shutil.rmtree")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.FileLock")
    def test_delete_dir_exception_oserror(
        self, mock_filelock, mock_rmtree, mock_exists
    ):
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError()

        cache = MMAPCache(cache_dir="/test/cache")
        result = cache._delete_dir("/test/dir")

        mock_rmtree.assert_called_once_with("/test/dir")
        self.assertFalse(result)

    def test_prune_cache_read_only_assertion(self):
        cache = MMAPCache(cache_dir="/test/cache", read_only=True)

        with self.assertRaises(AssertionError):
            cache.prune_cache(10)

    @patch.object(MMAPCache, "_remove_oldest")
    def test_prune_cache_no_checkpoint_frequency(self, mock_remove_oldest):
        cache = MMAPCache(
            cache_dir="/test/cache", read_only=False, model_checkpoint_frequency=None
        )
        cache.prune_cache(10)

        mock_remove_oldest.assert_called_once_with(10)

    @patch.object(MMAPCache, "_remove_oldest")
    def test_prune_cache_small_checkpoint_frequency(self, mock_remove_oldest):
        cache = MMAPCache(
            cache_dir="/test/cache",
            read_only=False,
            model_checkpoint_frequency=5,
            lookback_length=10,
        )
        cache.prune_cache(10)

        mock_remove_oldest.assert_called_once_with(10)

    @patch.object(MMAPCache, "get_content_indices")
    @patch.object(MMAPCache, "_delete_dir")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.makedirs")
    def test_prune_cache_init_empty_cache(
        self, mock_makedirs, mock_delete_dir, mock_get_indices
    ):
        mock_get_indices.return_value = []
        cache = MMAPCache(cache_dir="/test/cache", read_only=False)

        cache.prune_cache_init(5, 2)

        mock_delete_dir.assert_not_called()

    @patch.object(MMAPCache, "get_content_indices")
    @patch.object(MMAPCache, "_delete_dir")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.makedirs")
    def test_prune_cache_init_index_out_of_range(
        self, mock_makedirs, mock_delete_dir, mock_get_indices
    ):
        mock_get_indices.return_value = [10, 11, 12, 13, 14]
        cache = MMAPCache(cache_dir="/test/cache", read_only=False)

        cache.prune_cache_init(5, 2)

        mock_delete_dir.assert_called_once_with("/test/cache")
        self.assertEqual(mock_makedirs.call_count, 3)

    def test_prune_cache_init_read_only(self):
        cache = MMAPCache(cache_dir="/test/cache", read_only=True)

        with self.assertRaises(ValueError):
            cache.prune_cache_init(5, 2)

    def test_get_non_empty_values(self):
        cache_dict = {1: 5, 2: 0, 3: 10, 4: 0, 5: 3}
        result = MMAPCache.get_non_empty_values(cache_dict)
        self.assertEqual(result, [5, 10, 3])

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.torch.cuda.is_available"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.torch.cuda.current_device"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.dist.get_process_group_ranks"
    )
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.dist.all_gather")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.dist.get_world_size")
    @patch("torch.tensor")
    def test_all_gather_cache_size_warm_start(
        self,
        mock_tensor,
        mock_world_size,
        mock_all_gather,
        mock_get_ranks,
        mock_current_device,
        mock_cuda_available,
    ):
        mock_cuda_available.return_value = True
        mock_current_device.return_value = 0
        mock_get_ranks.return_value = [0, 1, 2, 3]
        mock_world_size.return_value = 4
        mock_cuda_available.return_value = False
        mock_tensor.return_value = Mock()

        def mock_all_gather_side_effect(tensor_list, cache_size_tensor, group):
            for i, tensor in enumerate(tensor_list):
                tensor.fill_(5)

        mock_all_gather.side_effect = mock_all_gather_side_effect

        cache = MMAPCache(cache_dir="/test/cache")
        mock_group = MagicMock()

        result_dict, restart_mode = cache.all_gather_cache_size(
            5, [0, 1, 2, 3], mock_group
        )

        expected_dict = {0: 5, 1: 5, 2: 5, 3: 5}
        self.assertEqual(result_dict, expected_dict)
        self.assertEqual(restart_mode, RestartMode.WARM_START)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.torch.cuda.is_available"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.torch.cuda.current_device"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.dist.get_process_group_ranks"
    )
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.dist.all_gather")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.dist.get_world_size")
    @patch("torch.tensor")
    def test_all_gather_cache_size_cold_start(
        self,
        mock_tensor,
        mock_world_size,
        mock_all_gather,
        mock_get_ranks,
        mock_current_device,
        mock_cuda_available,
    ):
        mock_cuda_available.return_value = True
        mock_current_device.return_value = 0
        mock_get_ranks.return_value = [0, 1, 2, 3]
        mock_world_size.return_value = 4
        mock_cuda_available.return_value = False
        mock_tensor.return_value = Mock()

        def mock_all_gather_side_effect(tensor_list, cache_size_tensor, group):
            values = [5, 0, 3, 0]
            for i, tensor in enumerate(tensor_list):
                tensor.fill_(values[i])

        mock_all_gather.side_effect = mock_all_gather_side_effect

        cache = MMAPCache(cache_dir="/test/cache")
        mock_group = MagicMock()

        result_dict, restart_mode = cache.all_gather_cache_size(
            5, [0, 1, 2, 3], mock_group
        )

        expected_dict = {0: 5, 1: 0, 2: 3, 3: 0}
        self.assertEqual(result_dict, expected_dict)
        self.assertEqual(restart_mode, RestartMode.COLD_START)

    @patch.object(MMAPCache, "_write_final")
    def test_set_final_index(self, mock_write_final):
        cache = MMAPCache(cache_dir="/test/cache")
        cache.set_final_index(42)

        mock_write_final.assert_called_once_with(42)

    @patch.object(MMAPCache, "create_staging_entry")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.rename")
    def test_write_final(self, mock_rename, mock_create_staging):
        mock_create_staging.return_value = "/tmp/staging_file"
        cache = MMAPCache(cache_dir="/test/cache")

        cache._write_final(42)

        mock_create_staging.assert_called_once()
        mock_rename.assert_called_once_with(
            "/tmp/staging_file", "/test/cache/complete/42"
        )

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    def test_is_final_index_no_complete_dir(self, mock_exists):
        mock_exists.return_value = False
        cache = MMAPCache(cache_dir="/test/cache")

        result = cache.is_final_index(42)
        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_is_final_index_empty_complete_dir(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = []
        cache = MMAPCache(cache_dir="/test/cache")

        result = cache.is_final_index(42)
        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_is_final_index_multiple_files_error(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["40", "41"]
        cache = MMAPCache(cache_dir="/test/cache")

        with self.assertRaises(ValueError):
            cache.is_final_index(42)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_is_final_index_true(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["40"]
        cache = MMAPCache(cache_dir="/test/cache")

        result = cache.is_final_index(42)
        self.assertTrue(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.listdir")
    def test_is_final_index_false(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["45"]
        cache = MMAPCache(cache_dir="/test/cache")

        result = cache.is_final_index(42)
        self.assertFalse(result)

    @patch.object(MMAPCache, "_delete_file")
    def test_remove_lookback_no_checkpoint_frequency(self, mock_delete_file):
        cache = MMAPCache(cache_dir="/test/cache", model_checkpoint_frequency=None)
        cache._remove_lookback(100)

        mock_delete_file.assert_not_called()

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch.object(MMAPCache, "_delete_file")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.threading.Thread")
    def test_remove_lookback_with_files(
        self, mock_thread, mock_delete_file, mock_exists
    ):
        mock_exists.side_effect = lambda path: path.endswith(
            "batch_100.pt"
        ) or path.endswith("batch_101.pt")
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        cache = MMAPCache(
            cache_dir="/test/cache", model_checkpoint_frequency=50, lookback_length=5
        )
        cache._remove_lookback(100)

        self.assertEqual(mock_thread.call_count, 2)
        self.assertEqual(mock_thread_instance.start.call_count, 2)
        self.assertEqual(mock_thread_instance.join.call_count, 2)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.os.path.exists")
    @patch.object(MMAPCache, "_delete_file")
    @patch("hyperpod_checkpointless_training.dataloader.mmap.cache.threading.Thread")
    def test_remove_lookback_exception_runtime(
        self, mock_thread, mock_delete_file, mock_exists
    ):
        mock_exists.side_effect = lambda path: path.endswith(
            "batch_100.pt"
        ) or path.endswith("batch_101.pt")
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.join.side_effect = RuntimeError()

        cache = MMAPCache(
            cache_dir="/test/cache", model_checkpoint_frequency=50, lookback_length=5
        )
        cache._remove_lookback(100)

        self.assertEqual(mock_thread.call_count, 2)
        self.assertEqual(mock_thread_instance.start.call_count, 2)
        # should fail on first join, and so only 1 join call
        self.assertEqual(mock_thread_instance.join.call_count, 1)

    def test_try_removing_cache_dir_read_only(self):
        cache = MMAPCache(
            cache_dir="/test/cache", read_only=True, force_cold_start=True
        )
        cache._try_removing_cache_dir()

    @patch.object(MMAPCache, "_delete_dir")
    def test_try_removing_cache_dir_force_cold_start(self, mock_delete_dir):
        mock_delete_dir.return_value = False
        cache = MMAPCache(
            cache_dir="/test/cache", read_only=False, force_cold_start=True
        )

        cache._try_removing_cache_dir()
        mock_delete_dir.assert_called_once_with("/test/cache")


class TestTTLMMAPCache(unittest.TestCase):

    def test_check_ttl(self):
        ttl_seconds = 1
        cache = TTLMMAPCache(
            ttl_seconds=ttl_seconds, cache_dir="/test/cache", read_only=True
        )
        # Create a test file
        test_file_path = "test_file.txt"
        with open(test_file_path, "w") as f:
            f.write("dummy content")

        # Before TTL expires, should return False
        self.assertFalse(cache.check_ttl(test_file_path))
        time.sleep(ttl_seconds + 1)
        # After TTL expires, should return True
        self.assertTrue(cache.check_ttl(test_file_path))

        # Clean up
        os.remove(test_file_path)
        # When file does not exist, should return False
        self.assertFalse(cache.check_ttl(test_file_path))

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.MMAPCache.get_content_indices",
        return_value=[0, 2, 4],
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.MMAPCache._delete_file"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.TTLMMAPCache.check_ttl",
        return_value=True,
    )
    def test_cleanup_expired_files_true(
        self, mock_check_ttl, mock_delete_file, mock_get_indices
    ):
        # test cleanup when files are expired
        cache = TTLMMAPCache(cache_dir="/test/cache")
        cache.cleanup_expired_files()
        mock_get_indices.assert_called_once()
        # all files should be checked and deleted
        self.assertEqual(mock_check_ttl.call_count, 3)
        self.assertEqual(mock_delete_file.call_count, 3)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.MMAPCache.get_content_indices",
        return_value=[0, 2, 4],
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.MMAPCache._delete_file"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache.TTLMMAPCache.check_ttl",
        return_value=False,
    )
    def test_cleanup_expired_files_false(
        self, mock_check_ttl, mock_delete_file, mock_get_indices
    ):
        # test cleanup when no files are expired
        cache = TTLMMAPCache(cache_dir="/test/cache")
        cache.cleanup_expired_files()
        mock_get_indices.assert_called_once()
        # only earliest file should be checked and rest skipped
        self.assertEqual(mock_check_ttl.call_count, 1)
        # no files should be deleted
        self.assertEqual(mock_delete_file.call_count, 0)

    def test_cleanup_expired_files_worker_read_only_true(self):
        cache = TTLMMAPCache(cache_dir="/test/cache", read_only=True)
        mock_stop_event = Mock()
        mock_stop_event.is_set.return_value = True
        cache.stop_event = mock_stop_event
        cache.cleanup_expired_files_worker()
        # Make sure exited early
        mock_stop_event.assert_not_called()

    @patch("threading.Thread")
    def test_cleanup_expired_files_worker_read_only_false(self, mock_thread):
        cache = TTLMMAPCache(cache_dir="/test/cache", read_only=False)
        cache.start_cleanup_expired_files_thread()
        # Make sure called once
        mock_thread.assert_called_once()

    @patch("threading.Thread")
    def test_start_cleanup_expired_files_thread_read_only_true(self, mock_thread):
        cache = TTLMMAPCache(cache_dir="/test/cache", read_only=True)
        cache.start_cleanup_expired_files_thread()
        # Make sure not called
        mock_thread.assert_not_called()

    def test_stop_cleanup_expired_files_thread(self):
        cache = TTLMMAPCache(cache_dir="/test/cache", read_only=False)
        mock_thread = Mock()
        cache.cleanup_thread = mock_thread
        cache.stop_event = Mock()

        cache.stop_cleanup_expired_files_thread()
        cache.stop_event.set.assert_called_once()
        mock_thread.join.assert_called_once()
        self.assertIsNone(cache.cleanup_thread)

    def test_def_ttl_does_not_change(self):
        # test that the default 1 hour TTL does not change
        default_ttl = 3600.0  # 1 hour
        threshold = 10
        base_path = "/tmp/test/cache"
        shutil.rmtree(base_path, ignore_errors=True)
        os.makedirs(f"{base_path}/tmp", exist_ok=True)

        cache = TTLMMAPCache(cache_dir=base_path, read_only=False)

        staging_entry_path = cache.stage_content("dummy content")
        cache.promote_content(staging_entry_path, 0)

        ready_entry_path = cache._get_batch_filename(0)

        # Current time
        base_time = time.time()

        # Before TTL expires, should return False
        with patch("time.time", return_value=base_time + default_ttl - threshold):
            self.assertFalse(cache.check_ttl(ready_entry_path))

        # Simulate time passing beyond TTL
        with patch("time.time", return_value=base_time + default_ttl + threshold):
            # After TTL expires, should return True
            self.assertTrue(cache.check_ttl(ready_entry_path))

            # After cleanup, file should be deleted
            cache.cleanup_expired_files()
            self.assertFalse(os.path.exists(ready_entry_path))

        shutil.rmtree(base_path, ignore_errors=True)

    def test_ttl_cleanup_e2e(self):
        # test cleanup end-to-end
        base_path = "/tmp/test/cache"
        shutil.rmtree(base_path, ignore_errors=True)
        os.makedirs(f"{base_path}/tmp", exist_ok=True)
        ttl_seconds = 1
        cache = TTLMMAPCache(
            ttl_seconds=ttl_seconds, cache_dir=base_path, read_only=False
        )
        cache.start_cleanup_expired_files_thread()

        NUM_ENTRIES = 2
        STEP_INTERVAL = 5
        staging_entry_paths = []
        for _ in range(NUM_ENTRIES):
            staging_entry_path = cache.stage_content("dummy content")
            staging_entry_paths.append(staging_entry_path)

        for i, staging_entry_path in enumerate(staging_entry_paths):
            cache.promote_content(staging_entry_path, i * STEP_INTERVAL)

        ready_entry_path = cache._get_batch_filename(0)

        time.sleep(0.2)
        # Check that before TTL is reached, file is not expired
        self.assertFalse(cache.check_ttl(ready_entry_path))
        time.sleep(ttl_seconds + 1)
        # Check that after TTL is reached, file is expired
        self.assertTrue(cache.check_ttl(ready_entry_path))
        cache.cleanup_expired_files()
        # Verify all files are removed when cleanup is called
        for i in range(len(staging_entry_paths)):
            entry_path = cache._get_batch_filename(i * STEP_INTERVAL)
            self.assertFalse(os.path.exists(entry_path))

        shutil.rmtree(base_path, ignore_errors=True)
