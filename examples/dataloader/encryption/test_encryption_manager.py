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

"""Example usage of the KMSEncryptionManager for encrypting and decrypting data."""

# Standard Library
import os

# Third Party
import torch

# First Party
from hyperpod_checkpointless_training.dataloader.encryption.encryption_manager import KMSEncryptionManager


def main():
    aws_kms_key = os.environ.get("AWS_KMS_KEY_ID", None)
    aws_region = os.environ.get("AWS_REGION", None)
    # If encryption dependencies are not installed, the following
    # should raise an ImportError.
    encryption_manager = KMSEncryptionManager(aws_kms_key, aws_region)

    data = {"a": torch.randn(2, 3), "b": [1, 2, 3], "c": "hello", "d": 42}
    print(f"Original: {data}")

    # Test encrypting the data
    encrypted = encryption_manager.encrypt(data)
    print(f"Encrypted: {encrypted}")

    # Test saving the encrypted data to a file
    encrypted_filepath = "encrypted.pt"
    torch.save(encrypted, encrypted_filepath)
    print(f"Encrypted data saved to {encrypted_filepath}")

    # Test directly decrypting the encrypted data
    decrypted = encryption_manager.decrypt(encrypted)
    print(f"Decrypted: {decrypted}")

    # Test decrypting the saved encrypted file
    load_encrypted = torch.load("encrypted.pt")
    load_decrypted = encryption_manager.decrypt(load_encrypted)
    print(f"Decrypted loaded file: {load_decrypted}")

    # Cleanup the saved encrypted file
    os.remove(encrypted_filepath)


if __name__ == "__main__":
    main()
