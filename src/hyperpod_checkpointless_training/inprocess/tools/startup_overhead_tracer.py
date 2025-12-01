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

import copy
import os
import threading
from typing import Optional

from lightning.pytorch.callbacks import Callback
from viztracer import VizTracer


# VizTracer constants
NUM_STEPS_TO_TRACE: int = int(os.environ.get("VIZTRACER_NUM_STEPS_TO_TRACE", 1))
TRACING_PROFILES_PATH: Optional[str] = os.environ.get("TRACING_PROFILES_PATH")
STARTUP_VIZTRACER_ENTRIES = int(os.environ.get("STARTUP_VIZTRACER_ENTRIES", "1000000"))
VIZTRACER_MIN_DURATION = int(os.environ.get("VIZTRACER_MIN_DURATION", "500"))


class StartupOverheadTracer(Callback):

    def __init__(self):
        # Exclude Triton and related files that cause compilation conflicts with VizTracer
        exclude_files = [
            "*/triton/*",
            "*/transformer_engine/*",
            "*/torch/_inductor/*",
        ]
        self.tracer = VizTracer(
            min_duration=VIZTRACER_MIN_DURATION, 
            tracer_entries=STARTUP_VIZTRACER_ENTRIES,
            exclude_files=exclude_files,
            ignore_c_function=True,
        )
        self.current_step = 0
        self.is_stopped = False

        self.tracer.start()
        print(f"VizTracer started with exclusions: {exclude_files}")

    def __deepcopy__(self, memo):
        # upgrades on the libraries are pickling objects for multiprocess work, so this is needed
        # https://tiny.amazon.com/12xzhf597/taskamazdevtaskad20

        # The deepcopy method is used to perform an io_dump. However, attempting this with
        # viztracer.VizTracer results in a pickling error. To enable serialization,
        # the class's default deepcopy behavior is overridden to exclude
        # the tracer attribute and any attributes that might contain ProcessGroup objects.

        cls = self.__class__
        result = cls.__new__(cls)

        memo[id(self)] = result

        if hasattr(self, "_cache"):
            memo[id(self._cache)] = self._cache.__new__(dict)

        # List of attributes to exclude from deep copying to avoid ProcessGroup pickling errors
        excluded_attrs = {"tracer"}
        
        for k, v in self.__dict__.items():
            if k in excluded_attrs:
                continue
                
            try:
                # Try to deep copy the attribute, but catch any pickling errors
                if callable(v):
                    setattr(result, k, v)
                else:
                    setattr(result, k, copy.deepcopy(v, memo))
            except (TypeError, AttributeError) as e:
                # If we can't pickle it (e.g., ProcessGroup objects), just copy the reference
                # or set to None for non-essential attributes
                if "cannot pickle" in str(e) or "ProcessGroup" in str(e):
                    # For simple types, copy the value directly
                    if isinstance(v, (int, float, str, bool, type(None))):
                        setattr(result, k, v)
                    else:
                        # For complex objects that can't be pickled, set to None
                        setattr(result, k, None)
                else:
                    # Re-raise other errors
                    raise

        # remove tracer from the deepcopied object to avoid issue with context dump
        setattr(result, "tracer", None)
        return result

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.is_stopped or self.tracer is None:
            return

        self.current_step += 1

        if self.current_step > NUM_STEPS_TO_TRACE:
            # avoid startup cost, import lazily
            from datetime import datetime, timezone
            from pathlib import Path

            try:
                self.tracer.stop()
                self.is_stopped = True
                print(f"VizTracer stopped (after tracing training setup and {NUM_STEPS_TO_TRACE} steps)")

                if TRACING_PROFILES_PATH is None:
                    print("TRACING_PROFILES_PATH not set, not writing any profile from viztracer")
                    return

                os.makedirs(TRACING_PROFILES_PATH, exist_ok=True)

                job_name = os.environ.get("JOB_NAME", "unknown")
                local_rank = os.environ.get("LOCAL_RANK", "unknown")
                world_size = os.environ.get("WORLD_SIZE", "unknown")
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                formed_filename = f"viztracer_startup_{job_name}_ws{world_size}_r{local_rank}_{timestamp}.json"
                save_filepath = str(Path(TRACING_PROFILES_PATH) / formed_filename)

                self.tracer.save(save_filepath)
            except Exception as e:
                print(f"Error stopping VizTracer: {e}")
                self.is_stopped = True
