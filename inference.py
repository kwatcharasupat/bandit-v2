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

import torchaudio as ta


@hydra.main(config_path="expt", config_name="inference")
def test(cfg: DictConfig):
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    system = build_system(cfg)

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
    else:
        raise ValueError("ckpt_path must be provided")

    audio, fs = ta.load(cfg.test_audio)
    print(audio.shape)

    if fs != cfg.fs:
        audio = ta.functional.resample(audio, fs, cfg.fs)

    batch = {
        "mixture": {
            "audio": audio[None, :, :].to("cuda"),
        }
    }

    system.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    system.to("cuda")
    system.eval()

    with torch.inference_mode():
        output = system.inference_handler(batch["mixture"]["audio"], system.model)

    if "output_path" not in cfg:
        output_path = os.path.join(
            os.path.dirname(cfg.test_audio), "estimates", cfg.model_variant
        )
    else:
        output_path = cfg.output_path

    os.makedirs(output_path, exist_ok=True)

    for stem in output["estimates"]:
        ta.save(
            os.path.join(output_path, f"{stem}_estimate.wav"),
            output["estimates"][stem]["audio"][0].cpu(),
            fs,
        )


if __name__ == "__main__":
    test()
