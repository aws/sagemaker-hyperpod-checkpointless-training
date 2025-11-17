"""KMS Encryption Manager."""

# Standard Library
import io
import os
from typing import Any

# Third Party
import boto3
import torch

try:
    # Keep encryption imports in try block to allow for optional dependency
    import aws_encryption_sdk
    from aws_cryptographic_material_providers.mpl import (
        AwsCryptographicMaterialProviders,
    )
    from aws_cryptographic_material_providers.mpl.config import MaterialProvidersConfig
    from aws_cryptographic_material_providers.mpl.models import CreateAwsKmsKeyringInput
    from aws_cryptographic_material_providers.mpl.references import IKeyring
    from aws_encryption_sdk import CommitmentPolicy

    HAVE_ENCRYPTION = True
except ImportError:
    HAVE_ENCRYPTION = False


if HAVE_ENCRYPTION:

    class KMSEncryptionManager:
        """
        KMS Encryption Manager.

        Supports local encryption/decryption using AWS KMS Key with AWS Encryption SDK.

        ref:
        https://docs.aws.amazon.com/encryption-sdk/latest/developer-guide/python-example-code.html
        """

        def __init__(
            self, kms_key_id: str | None = None, region: str | None = None
        ) -> None:
            """
            Initialize the KMS Encryption Manager.

            Args:
                kms_key_id: The KMS Key ID.
                    Will fallback to AWS_KMS_KEY_ID in the environment if not provided.
                region: The AWS region of the KMS Key.
                    Will fallback to AWS_REGION in the environment if not provided.
            """
            if kms_key_id is None:
                kms_key_id = os.environ.get("AWS_KMS_KEY_ID", None)
            self.kms_key_id = kms_key_id
            if region is None:
                region = os.environ.get("AWS_REGION", None)
            self.region = region
            if self.kms_key_id is None:
                raise ValueError(
                    "Key ID cannot be None. Please pass in a valid KMS Key ID "
                    "or set AWS_KMS_KEY_ID in the environment."
                )
            if self.region is None:
                raise ValueError(
                    "Region cannot be None. Please pass in a valid AWS region "
                    "or set AWS_REGION in the environment."
                )
            self.kms_client = self._create_kms_client(self.region)
            self.kms_keyring = self._create_kms_keyring(
                self.kms_client, self.kms_key_id
            )
            self.encryption_client = aws_encryption_sdk.EncryptionSDKClient(
                commitment_policy=CommitmentPolicy.REQUIRE_ENCRYPT_REQUIRE_DECRYPT
            )

        def _create_kms_client(self, region: str):
            """Create a boto3 KMS client."""
            kms_client = boto3.client("kms", region)
            return kms_client

        def _create_kms_keyring(self, kms_client, kms_key_id: str) -> IKeyring:
            """Create an AWS KMS keyring."""
            mat_prov: AwsCryptographicMaterialProviders = (
                AwsCryptographicMaterialProviders(config=MaterialProvidersConfig())
            )

            keyring_input: CreateAwsKmsKeyringInput = CreateAwsKmsKeyringInput(
                kms_key_id=kms_key_id, kms_client=kms_client
            )

            kms_keyring: IKeyring = mat_prov.create_aws_kms_keyring(input=keyring_input)
            return kms_keyring

        def encrypt(self, data: Any) -> bytes:
            """Encrypt the data."""
            serialized = self.serialize(data)
            ciphertext = self._encrypt(serialized)
            return ciphertext

        def decrypt(self, ciphertext: bytes) -> Any:
            """Decrypt the ciphertext."""
            plaintext = self._decrypt(ciphertext)
            data = self.deserialize(plaintext)
            return data

        def _encrypt(self, plaintext: bytes) -> bytes:
            """Encrypt plaintext bytes with KMS Keyring."""
            # Generates new data key each time by default
            ciphertext, _ = self.encryption_client.encrypt(
                source=plaintext,
                keyring=self.kms_keyring,
            )
            return ciphertext

        def _decrypt(self, ciphertext: bytes) -> bytes:
            """Decrypt ciphertext with KMS Keyring."""
            # Uses encrypted data key in ciphertext to decrypt
            plaintext, _ = self.encryption_client.decrypt(
                source=ciphertext,
                keyring=self.kms_keyring,
            )
            return plaintext

        @staticmethod
        def serialize(data) -> bytes:
            """Serialize the data."""
            buffer = io.BytesIO()
            torch.save(data, buffer)
            buffer.seek(0)
            return buffer.read()

        @staticmethod
        def deserialize(byte_string: bytes) -> Any:
            """Deserialize the byte string."""
            return torch.load(io.BytesIO(byte_string))

else:

    class KMSEncryptionManager:
        """
        Fallback stub when aws-encryption-sdk[MPL] is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:
            """
            Raise ImportError when instantiated.
            """
            raise ImportError(
                "Batch encryption support with KMSEncryptionManager requires "
                "the optional dependency aws-encryption-sdk[MPL]. "
                "Please install with `pip install aws-encryption-sdk[MPL]>=4`."
            )
