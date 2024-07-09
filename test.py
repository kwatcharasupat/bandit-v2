import os
import hydra
import pytorch_lightning as pl

# from ray.utils.accelerators import NVIDIA_A100
import torch
from omegaconf import DictConfig, OmegaConf

from src.system.utils import build_datamodule, build_system, build_trainer

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(config_path="expt", config_name="default")
def test(cfg: DictConfig):
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    cfg.data.datamodule.kwargs.batch_size_per_gpu = 1

    datamodule = build_datamodule(cfg, test_only=True)

    system = build_system(cfg)

    trainer = build_trainer(cfg, use_ray=False)

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
    else:
        raise ValueError("ckpt_path must be provided")

    # trainer.logger.save()

    if trainer.is_global_zero:
        log_dir = trainer.log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))

    trainer.test(system, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    test()
