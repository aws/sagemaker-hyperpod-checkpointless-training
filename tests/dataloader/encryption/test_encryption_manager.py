"""Test KMS Encryption Manager."""

# Standard Library
import unittest
from unittest.mock import Mock, patch

# Third Party
import torch

# First Party
from hyperpod_checkpointless_training.dataloader.encryption.encryption_manager import KMSEncryptionManager

try:
    # Keep encryption imports in try block to allow for optional dependency
    import aws_encryption_sdk
    from aws_cryptographic_material_providers.mpl import (
        AwsCryptographicMaterialProviders,
    )

    HAVE_ENCRYPTION = True
except ImportError:
    HAVE_ENCRYPTION = False


@unittest.skipIf(
    not HAVE_ENCRYPTION, "encryption dependencies not installed, skipping tests"
)
class TestKMSEncryptionManager(unittest.TestCase):
    """Test KMS Encryption Manager."""

    def setUp(self):
        """Setup KMS Encryption Manager for tests."""
        self.encryption_manager = self._create_encryption_manager()

    def _create_encryption_manager(
        self, kms_key_id: str = "test-key-id", region: str = "placeholder-region"
    ) -> KMSEncryptionManager:
        """Create KMS Encryption Manager."""
        return KMSEncryptionManager(kms_key_id, region)

    def _compare_dicts(self, dict1: dict, dict2: dict) -> None:
        """Compare two dictionaries for equality."""
        self.assertEqual(
            dict1.keys(),
            dict2.keys(),
            f"Mismatch in keys. {dict1.keys()=} and {dict2.keys()=}.",
        )

        for key in dict1:
            if isinstance(dict1[key], torch.Tensor):
                self.assertTrue(
                    torch.equal(dict1[key], dict2[key]),
                    f"Mismatch on key: {key}. {dict1[key]=} and {dict2[key]=}.",
                )
            else:
                self.assertEqual(
                    dict1[key],
                    dict2[key],
                    f"Mismatch on key: {key}. {dict1[key]=} and {dict2[key]=}.",
                )

    @patch("boto3.client")
    @patch(
        "aws_cryptographic_material_providers.mpl.AwsCryptographicMaterialProviders"
        ".create_aws_kms_keyring"
    )
    def test_encryption_manager_init(self, mock_create_keyring, mock_boto_client):
        """Test initialization of KMSEncryptionManager."""
        encryption_manager = self._create_encryption_manager()

        self.assertIsNotNone(encryption_manager)
        mock_boto_client.assert_called_once()
        mock_create_keyring.assert_called_once()

    @patch("boto3.client")
    def test_create_kms_client(self, mock_boto_client):
        """Test creation of KMS client."""
        region = "us-west-2"
        client = self.encryption_manager._create_kms_client(region)

        self.assertIsNotNone(client)
        mock_boto_client.assert_called_once_with("kms", region)

    def test_create_kms_keyring(self):
        """Test creation of KMS keyring."""
        mock_kms_client = Mock()
        kms_keyring = self.encryption_manager._create_kms_keyring(
            mock_kms_client, "test-key-id"
        )
        self.assertIsNotNone(kms_keyring)

    def test_serialize_deserialize(self):
        """Test serialization and deserialization of data."""
        data = {"a": torch.randn(4, 2), "b": [7, 7, 7], "c": "hello", "d": 42}
        serialized = KMSEncryptionManager.serialize(data)

        self.assertIsInstance(serialized, bytes)
        self.assertNotEqual(serialized, data)

        deserialized = KMSEncryptionManager.deserialize(serialized)

        self._compare_dicts(data, deserialized)

    @patch(
        "aws_encryption_sdk.EncryptionSDKClient.encrypt",
        return_value=(b"encrypted-data", None),
    )
    def test_encryption_client_encrypt(self, mock_encrypt):
        """Test encryption client encrypt call."""
        data = b"i-am-a-byte-string"
        expected_encrypted_data = b"encrypted-data"
        encrypted_data = self.encryption_manager._encrypt(data)
        self.assertIsNotNone(encrypted_data)
        self.assertNotEqual(encrypted_data, data)
        mock_encrypt.assert_called_once()
        self.assertEqual(encrypted_data, expected_encrypted_data)

    @patch(
        "aws_encryption_sdk.EncryptionSDKClient.decrypt",
        return_value=(b"i-am-a-byte-string", None),
    )
    def test_encryption_client_decrypt(self, mock_decrypt):
        """Test encryption client decrypt call."""
        expected_data = b"i-am-a-byte-string"
        encrypted_data = b"encrypted-data"
        decrypted_data = self.encryption_manager._decrypt(encrypted_data)
        self.assertIsNotNone(decrypted_data)
        mock_decrypt.assert_called_once()
        self.assertEqual(decrypted_data, expected_data)

    def test_e2e_encrypt_decrypt(self):
        """Test end-to-end encryption and decryption of data."""
        data = {
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "b": [7, 7, 7],
            "c": "hello",
            "d": 42,
        }
        # assume serialization is okay
        serialized_data = KMSEncryptionManager.serialize(data)

        with patch(
            "aws_encryption_sdk.EncryptionSDKClient.encrypt",
            return_value=(b"encrypted-data", None),
        ) as mock_encrypt:
            encrypted_data = self.encryption_manager.encrypt(data)
            self.assertIsNotNone(encrypted_data)
            mock_encrypt.assert_called_once()

        with patch(
            "aws_encryption_sdk.EncryptionSDKClient.decrypt",
            return_value=(serialized_data, None),
        ) as mock_decrypt:
            decrypted_data = self.encryption_manager.decrypt(encrypted_data)
            self.assertIsNotNone(decrypted_data)
            mock_decrypt.assert_called_once()

        self._compare_dicts(data, decrypted_data)
