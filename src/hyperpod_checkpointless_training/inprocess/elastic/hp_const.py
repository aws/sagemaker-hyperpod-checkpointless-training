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

import struct

HEADER_TYPE = "!I"
HEADER_SIZE = struct.calcsize(HEADER_TYPE)

# store prefix
HP_AGENT_BARRIER_PREFIX = "hyperpod/agent/barrier"
HP_AGENT_FAILURE_PREFIX = "hyperpod/agent/failure"
HP_AGENT_STORE_BARRIER_PREFIX = "hyperpod/agent/store/barrier"
