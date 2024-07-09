import logging

import hydra
import pytorch_lightning as pl
import ray

# from ray.utils.accelerators import NVIDIA_A100
import torch
from omegaconf import DictConfig
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from src.system.utils import build_datamodule, build_system, build_trainer

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


# @ray.remote(num_gpus=1, accelerator_type=NVIDIA_A100)
def train_func_per_worker(cfg):
    torch.set_float32_matmul_precision("high")
    cudnn_benchmark = True  # noqa: F841
    cudnn_allow_tf32 = True  # noqa: F841

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # for logger in loggers:
    #     logger.setLevel(logging.DEBUG)

    datamodule = build_datamodule(cfg)

    system = build_system(cfg)

    world_rank = ray.train.get_context().get_world_rank()
    print(f"World Rank: {world_rank}")

    trainer = build_trainer(cfg, world_rank=world_rank)

    # if world_rank == 0:
    #     if trainer.logger is not None:
    #         trainer.logger.save()
    #         log_dir = trainer.log_dir

    #         if log_dir is not None:
    #             os.makedirs(log_dir, exist_ok=True)
    #             OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
    else:
        print("No checkpoint path provided")

    print("Beginning training")
    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.get("run_test", False):
        trainer.test(system, datamodule=datamodule)


@hydra.main(config_path="expt", config_name="default")
def train(cfg: DictConfig):
    print("Entering train function")

    # ray.init(resources={"accelerator_type": "A100"})
    ray.init(
        # runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}},
        configure_logging=True,
        logging_level=logging.INFO,
    )

    cluster_name = cfg.ray.get("cluster_name", "workbench")

    if cluster_name == "workbench":
        resources_per_worker = {"GPU": 1, "CPU": 10}
        kwargs = {}
    elif cluster_name == "genai":
        resources_per_worker = {
            "GPU": 1,
            "CPU": 8,
            "accelerator_type:A100": 0.01,
        }  # "gpu_type_A100": 1.0}
        kwargs = {}
    else:
        raise ValueError(f"Unknown cluster name: {cluster_name}")

    scaling_config = ScalingConfig(
        num_workers=cfg.ray.num_workers,
        use_gpu=cfg.ray.use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="STRICT_PACK",
        **kwargs,
    )

    run_config = RunConfig(
        storage_path=cfg.ray.storage_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=10,
        ),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    train()
