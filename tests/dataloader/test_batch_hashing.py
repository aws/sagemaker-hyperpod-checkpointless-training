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
"""Unit tests for batch hashing functionality with fake dataset."""

import unittest
import torch
import numpy as np

from hyperpod_checkpointless_training.dataloader.batch_hashing import (
    _create_checksum_str,
    tensor_to_hash,
    batch_to_hash,
    compute_batch_with_hash,
)


class TestFakeDatasetBatchHashing(unittest.TestCase):
    """Test batch hashing functionality with synthetic data."""

    def setUp(self):
        """Set up test fixtures with fake data."""
        # Set random seed for reproducible tests
        torch.manual_seed(42)
        np.random.seed(42)

        # Create fake tensor data of various types and shapes
        self.fake_tensor_float32 = torch.randn(4, 8, dtype=torch.float32)
        self.fake_tensor_bfloat16 = torch.randn(2, 4, dtype=torch.bfloat16)
        self.fake_tensor_int64 = torch.randint(0, 100, (3, 5), dtype=torch.int64)

        # Create fake batch structures similar to training data
        self.fake_simple_batch = {
            "input_ids": torch.randint(0, 1000, (2, 10), dtype=torch.int64),
            "attention_mask": torch.ones(2, 10, dtype=torch.int64),
            "labels": torch.randint(0, 1000, (2, 10), dtype=torch.int64),
        }

        self.fake_complex_batch = {
            "tokens": {
                "input_ids": torch.randint(0, 1000, (4, 20), dtype=torch.int64),
                "position_ids": torch.arange(20).unsqueeze(0).repeat(4, 1),
            },
            "metadata": {
                "seq_length": 20,
                "batch_size": 4,
            },
            "loss_mask": torch.ones(4, 20, dtype=torch.float32),
        }

        self.fake_list_batch = [
            torch.randn(2, 5),
            torch.randint(0, 10, (3, 4)),
            {"nested": torch.ones(1, 3)},
        ]

    def test_tensor_hashing_consistency(self):
        """Test that identical tensors produce identical hashes."""
        # Test float32 tensor
        hash1 = tensor_to_hash(self.fake_tensor_float32)
        hash2 = tensor_to_hash(self.fake_tensor_float32)
        self.assertEqual(
            hash1, hash2, "Float32 tensor should produce consistent hashes"
        )

        # Test bfloat16 tensor (should be converted to float)
        hash1 = tensor_to_hash(self.fake_tensor_bfloat16)
        hash2 = tensor_to_hash(self.fake_tensor_bfloat16)
        self.assertEqual(
            hash1, hash2, "Bfloat16 tensor should produce consistent hashes"
        )

        # Test int64 tensor
        hash1 = tensor_to_hash(self.fake_tensor_int64)
        hash2 = tensor_to_hash(self.fake_tensor_int64)
        self.assertEqual(hash1, hash2, "Int64 tensor should produce consistent hashes")

    def test_tensor_hashing_different_values(self):
        """Test that different tensors produce different hashes."""
        tensor1 = torch.randn(4, 8, dtype=torch.float32)
        tensor2 = torch.randn(4, 8, dtype=torch.float32)

        hash1 = tensor_to_hash(tensor1)
        hash2 = tensor_to_hash(tensor2)
        self.assertNotEqual(
            hash1, hash2, "Different tensors should produce different hashes"
        )

    def test_tensor_hashing_different_shapes(self):
        """Test that tensors with different shapes produce different hashes."""
        tensor1 = torch.ones(2, 4)
        tensor2 = torch.ones(4, 2)

        hash1 = tensor_to_hash(tensor1)
        hash2 = tensor_to_hash(tensor2)
        self.assertNotEqual(
            hash1,
            hash2,
            "Tensors with different shapes should produce different hashes",
        )

    def test_batch_hashing_simple_dict(self):
        """Test batch hashing with simple dictionary structure."""
        hash1 = batch_to_hash(self.fake_simple_batch)
        hash2 = batch_to_hash(self.fake_simple_batch)
        self.assertEqual(
            hash1, hash2, "Simple batch dict should produce consistent hashes"
        )

        # Test that hash is a valid hex string
        self.assertIsInstance(hash1, str)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1.lower()))

    def test_batch_hashing_complex_nested_dict(self):
        """Test batch hashing with complex nested dictionary structure."""
        hash1 = batch_to_hash(self.fake_complex_batch)
        hash2 = batch_to_hash(self.fake_complex_batch)
        self.assertEqual(
            hash1, hash2, "Complex nested batch should produce consistent hashes"
        )

    def test_batch_hashing_list_structure(self):
        """Test batch hashing with list structure."""
        hash1 = batch_to_hash(self.fake_list_batch)
        hash2 = batch_to_hash(self.fake_list_batch)
        self.assertEqual(hash1, hash2, "List batch should produce consistent hashes")

    def test_batch_hashing_empty_structures(self):
        """Test batch hashing with empty structures."""
        empty_dict = {}
        empty_list = []

        hash_dict1 = batch_to_hash(empty_dict)
        hash_dict2 = batch_to_hash(empty_dict)
        self.assertEqual(
            hash_dict1, hash_dict2, "Empty dict should produce consistent hashes"
        )

        hash_list1 = batch_to_hash(empty_list)
        hash_list2 = batch_to_hash(empty_list)
        self.assertEqual(
            hash_list1, hash_list2, "Empty list should produce consistent hashes"
        )

        # Empty dict and empty list should produce different hashes
        self.assertNotEqual(
            hash_dict1,
            hash_list1,
            "Empty dict and list should produce different hashes",
        )

    def test_batch_hashing_with_prefix(self):
        """Test batch hashing with prefix parameter."""
        hash_no_prefix = batch_to_hash(self.fake_simple_batch)
        hash_with_prefix = batch_to_hash(self.fake_simple_batch, prefix="test")

        # Should be different due to prefix
        self.assertNotEqual(
            hash_no_prefix, hash_with_prefix, "Prefix should change the hash"
        )

        # Same prefix should produce same hash
        hash_with_prefix2 = batch_to_hash(self.fake_simple_batch, prefix="test")
        self.assertEqual(
            hash_with_prefix, hash_with_prefix2, "Same prefix should produce same hash"
        )

    def test_batch_hashing_scalar_values(self):
        """Test batch hashing with scalar values."""
        scalar_batch = {
            "learning_rate": 0.001,
            "epoch": 5,
            "step": 1000,
        }

        hash1 = batch_to_hash(scalar_batch)
        hash2 = batch_to_hash(scalar_batch)
        self.assertEqual(hash1, hash2, "Scalar batch should produce consistent hashes")

    def test_compute_batch_with_hash_consistency(self):
        """Test compute_batch_with_hash produces consistent results."""
        step = 100

        hash1 = compute_batch_with_hash(self.fake_simple_batch, step)
        hash2 = compute_batch_with_hash(self.fake_simple_batch, step)
        self.assertEqual(hash1, hash2, "Same batch and step should produce same hash")

    def test_compute_batch_with_hash_different_steps(self):
        """Test that different steps produce different hashes for same batch."""
        step1 = 100
        step2 = 101

        hash1 = compute_batch_with_hash(self.fake_simple_batch, step1)
        hash2 = compute_batch_with_hash(self.fake_simple_batch, step2)
        self.assertNotEqual(
            hash1, hash2, "Different steps should produce different hashes"
        )

    def test_compute_batch_with_hash_different_batches(self):
        """Test that different batches produce different hashes for same step."""
        step = 100

        batch1 = {"data": torch.randn(2, 3)}
        batch2 = {"data": torch.randn(2, 3)}

        hash1 = compute_batch_with_hash(batch1, step)
        hash2 = compute_batch_with_hash(batch2, step)
        self.assertNotEqual(
            hash1, hash2, "Different batches should produce different hashes"
        )

    def test_batch_hashing_mixed_types(self):
        """Test batch hashing with mixed data types."""
        mixed_batch = {
            "tensors": {
                "float_tensor": torch.randn(2, 3, dtype=torch.float32),
                "int_tensor": torch.randint(0, 10, (2, 3), dtype=torch.int64),
                "bool_tensor": torch.randint(0, 2, (2, 3), dtype=torch.bool),
            },
            "scalars": {
                "float_val": 3.14,
                "int_val": 42,
                "str_val": "test_string",
            },
            "lists": [1, 2, 3, torch.ones(2, 2)],
        }

        hash1 = batch_to_hash(mixed_batch)
        hash2 = batch_to_hash(mixed_batch)
        self.assertEqual(
            hash1, hash2, "Mixed type batch should produce consistent hashes"
        )

    def test_large_tensor_hashing(self):
        """Test hashing of large tensors."""
        large_tensor = torch.randn(100, 100, 50, dtype=torch.float32)

        hash1 = tensor_to_hash(large_tensor)
        hash2 = tensor_to_hash(large_tensor)
        self.assertEqual(hash1, hash2, "Large tensor should produce consistent hashes")

    def test_bfloat16_conversion_in_hashing(self):
        """Test that bfloat16 tensors are properly converted before hashing."""
        # Create identical data in different formats
        data = torch.randn(3, 4, dtype=torch.float32)
        tensor_float32 = data.clone()
        tensor_bfloat16 = data.to(torch.bfloat16)

        hash_float32 = tensor_to_hash(tensor_float32)
        hash_bfloat16 = tensor_to_hash(tensor_bfloat16)

        # They should be different due to precision loss in bfloat16 conversion
        # but the bfloat16 hash should be consistent
        hash_bfloat16_2 = tensor_to_hash(tensor_bfloat16)
        self.assertEqual(
            hash_bfloat16,
            hash_bfloat16_2,
            "Bfloat16 tensor should produce consistent hashes",
        )

    def test_create_checksum_str_functionality(self):
        """Test the underlying checksum creation function."""
        test_data = b"test_data_for_hashing"

        # Test with default MD5
        checksum1 = _create_checksum_str(test_data)
        checksum2 = _create_checksum_str(test_data)
        self.assertEqual(checksum1, checksum2, "Same data should produce same checksum")

        # Test with different data
        different_data = b"different_test_data"
        checksum3 = _create_checksum_str(different_data)
        self.assertNotEqual(
            checksum1, checksum3, "Different data should produce different checksums"
        )

    def test_deterministic_ordering_in_dict_hashing(self):
        """Test that dictionary key ordering doesn't affect hash (keys should be sorted)."""
        dict1 = {"c": torch.ones(2, 2), "a": torch.zeros(2, 2), "b": torch.randn(2, 2)}
        dict2 = {"a": torch.zeros(2, 2), "b": torch.randn(2, 2), "c": torch.ones(2, 2)}

        # Set the same random values for 'b' key to ensure identical content
        torch.manual_seed(123)
        dict1["b"] = torch.randn(2, 2)
        torch.manual_seed(123)
        dict2["b"] = torch.randn(2, 2)

        hash1 = batch_to_hash(dict1)
        hash2 = batch_to_hash(dict2)
        self.assertEqual(hash1, hash2, "Dictionary key order should not affect hash")


if __name__ == "__main__":
    unittest.main()
