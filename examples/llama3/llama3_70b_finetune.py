# Standard Library
import os

# Viztracer
tracer = None
if os.environ.get("ENABLE_VIZTRACER", "0") == "1" and os.environ.get("RANK", "0") == "0":
    if "TRACING_PROFILES_PATH" in os.environ:
        from hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer import StartupOverheadTracer

        tracer = StartupOverheadTracer()
    else:
        raise ValueError("ENABLE_VIZTRACER is set but TRACING_PROFILES_PATH is missing")

import logging
from functools import partial

# Third Party
from nemo.collections import llm
from nemo import lightning as nl
from nemo.lightning.pytorch.strategies import MegatronStrategy
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from nemo.collections.llm.gpt.model.llama import Llama3Config70B, LlamaModel

from megatron.core.distributed import DistributedDataParallelConfig

from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP

# First Party
from examples.dataloader.data_module import LLMDataModule


logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)

if tracer is not None:
    # printed here so it's processed off the main thread
    logger.info("Startup Overhead VizTracer started")


class TrainerBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def _validate(self):
        moe_expert_parallel = self.cfg.strategy.get("moe_expert_parallel", True)
        num_moe_experts = self.cfg.strategy.get("num_moe_experts", 1)
        if not moe_expert_parallel or num_moe_experts <= 1:
            return
        world_size = int(os.environ["WORLD_SIZE"])
        tensor_parallel_size = self.cfg.strategy.tensor_model_parallel_size
        pipeline_parallel_size = self.cfg.strategy.pipeline_model_parallel_size
        data_parallel_size = world_size // (
            tensor_parallel_size * pipeline_parallel_size
        )
        if (
            self.cfg.strategy.num_moe_experts % data_parallel_size != 0
            and data_parallel_size % num_moe_experts != 0
        ):
            raise ValueError(
                f"Data parallel size ({data_parallel_size}) must be divisible"
                f"by the number of experts ({num_moe_experts}) or vice versa"
            )

    @property
    def datamodule(self):

        base_data_module = LLMDataModule(
            dataset_path=self.cfg.dataset.dataset_path,
            micro_batch_size=self.cfg.data.micro_batch_size,
            global_batch_size=self.cfg.data.global_batch_size,
            seq_length=self.cfg.data.seq_length,
            num_workers=self.cfg.dataset.num_workers,
            partition=self.cfg.dataset.partition,
            pin_memory=self.cfg.dataset.get("pin_memory", False),
            shuffle=self.cfg.dataset.get("shuffle", False),
            drop_last=self.cfg.dataset.get("drop_last", True),
            keep_in_memory=self.cfg.dataset.get("keep_in_memory", False),
        )

        return base_data_module

    @property
    def callbacks(self):
        return list(instantiate(self.cfg.callbacks))

    @property
    def plugins(self):
        return [instantiate(self.cfg.plugins)]

    @property
    def ddp(self):
        return DistributedDataParallelConfig(**self.cfg.ddp)

    @property
    def strategy(self):
        return MegatronStrategy(**self.cfg.strategy, ddp=self.ddp)

    @property
    def model(self):
        model_cfg = OmegaConf.to_container(self.cfg.model.config, resolve=True)
        model_config = Llama3Config70B(**model_cfg)
        model = LlamaModel(model_config)
        return model

    @property
    def optim(self):
        cfg = instantiate(self.cfg.optim.config)
        lr_scheduler = instantiate(self.cfg.optim.lr_scheduler)
        return nl.MegatronOptimizerModule(config=cfg, lr_scheduler=lr_scheduler)

    def create_trainer(self):
        logging.info(self)
        self._validate()

        # Get callbacks and add tracer if needed
        callbacks = self.callbacks
        if tracer:
            callbacks.append(tracer)  # Add tracer callback only once before creating trainer

        return nl.Trainer(
            **self.cfg.trainer,
            callbacks=callbacks,
            strategy=self.strategy,
            plugins=self.plugins,
            logger=None,
        )

    def __repr__(self):
        cfg = self.cfg
        msg = f"world_size={os.environ['WORLD_SIZE']}\n"
        msg += f"tensor_parallel_size={cfg.strategy.tensor_model_parallel_size}\n"
        msg += f"pipeline_parallel_size={cfg.strategy.pipeline_model_parallel_size}\n"
        msg += f"moe_expert_parallel={cfg.strategy.get('moe_expert_parallel', True)}"
        return msg

    def __str__(self):
        return self.__repr__()


@hydra.main(
    version_base=None, config_path="./config/", config_name="llama3_70b_finetune.yaml"
)
def main(cfg):
    if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        logging.info("\n\n************** Experiment configuration ***********")
        logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    builder = TrainerBuilder(cfg)
    model = builder.model
    optim = builder.optim
    trainer = builder.create_trainer()
    datamodule = builder.datamodule

    peft = llm.peft.LoRA(dim=8, alpha=16, target_modules=['linear_qkv'])
    resume = instantiate(cfg.resume)
    llm.finetune(
        model=model,
        optim=optim,
        data=datamodule,
        trainer=trainer,
        peft=peft,
        resume=resume,
        log=instantiate(cfg.log),
    )


if __name__ == "__main__":
    main()
    # With python 3.12 we are noticing a seg fault at sys.exit when viztracer is enabled, leading to PLR
    if tracer is not None:
        os._exit(0)
