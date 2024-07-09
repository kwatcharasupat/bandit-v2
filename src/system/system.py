import json
import os
import pytorch_lightning as pl
import torch
import torch.distributed


class System(pl.LightningModule):
    def __init__(
        self, model, loss_handler, metric_handler, optim_handler, inference_handler, cfg
    ):
        super().__init__()

        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optim_handler = optim_handler
        self.inference_handler = inference_handler

        self.cfg = cfg

        self.strict_loading = False

    def setup(self, stage):
        print("Setting up system")
        return

    def prepare_data(self):
        print("Preparing data")
        return

    def set_rank(self):
        try:
            self.rank = self.trainer.global_rank
        except Exception as e:
            self.rank = 0
            print(f"Error setting rank: {e}")

        if self.inference_handler is not None:
            self.inference_handler.set_rank(self.rank)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, mode):
        # print(f"Common step for {mode}")
        batch = self.model(batch)

        loss_dict = self.compute_loss(batch)

        self.metric_handler.update(batch, mode)

        return loss_dict

    def training_step(self, batch, batch_idx):
        # print("Training step")
        loss_dict = self.common_step(batch, "train")
        # print("Loss dict", loss_dict)

        self.log_metrics(loss_dict["components"], "train/loss", prog_bar=True)
        self.log_metrics({"loss/total": loss_dict["total"]}, "train", prog_bar=True)

        return loss_dict["total"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, "val")
        self.log_metrics(loss_dict["components"], "val/loss")
        self.log_metrics({"loss/total": loss_dict["total"]}, "val", prog_bar=True)

        return loss_dict["total"]

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        assert (
            self.inference_handler is not None
        ), "Inference handler must be provided for testing"
        output = self.inference_handler(batch["mixture"]["audio"], self.model)
        batch["estimates"] = output["estimates"]

        batch_metrics = self.metric_handler.update_and_compute_batch(batch, "test")
        files = batch["identifier"]["file"]

        log_dir = os.path.join(self.trainer.log_dir, "test")
        os.makedirs(log_dir, exist_ok=True)

        for i in range(len(files)):
            metrics = {k: float(v[i]) for k, v in batch_metrics.items()}
            metrics["file"] = files[i]

            with open(os.path.join(log_dir, f"{files[i]}.json"), "w") as f:
                json.dump(metrics, f)

        return metrics

    @torch.no_grad()
    def inference_step(self, batch, batch_idx):
        assert (
            self.inference_handler is not None
        ), "Inference handler must be provided for inference"
        output = self.inference_handler(batch["mixture"]["audio"], self.model)
        batch["estimates"] = output["estimates"]

        return batch

    def predict_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def compute_loss(self, batch):
        return self.loss_handler(batch)

    def configure_optimizers(self):
        return self.optim_handler.configure_optimizers(self.model)

    def log_metrics(self, metrics, mode, prog_bar=False):
        metrics = {f"{mode}/{k}": v for k, v in metrics.items()}
        on_step = True if "train" in mode else False
        on_epoch = True if "train" not in mode else False

        self.log_dict(
            metrics,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=True,
        )

    def on_train_epoch_start(self):
        print("Train epoch starts")
        self.metric_handler.reset("train")
        print("Train metrics reset")

    def on_validation_epoch_start(self):
        print("Val epoch starts")
        self.metric_handler.reset("val")

    def on_test_epoch_start(self):
        self.set_rank()
        self.metric_handler.reset("test")

    def on_inference_epoch_start(self):
        self.metric_handler.reset("inference")

    def on_train_epoch_end(self):
        self.metric_handler.reset("train")
        print("Train epoch end")

    def on_validation_epoch_end(self):
        print("Computing metrics for validation set")
        metrics = self.metric_handler.compute("val")
        print("Logging metrics for validation set")
        self.log_metrics(metrics, "val")
        self.metric_handler.reset("val")
        
        print(metrics)
        
        print("Val epoch end")

    def on_test_epoch_end(self):
        metrics = self.metric_handler.compute("test")
        self.log_metrics(metrics, "test")
        self.metric_handler.reset("test")

    def on_inference_epoch_end(self):
        metrics = self.metric_handler.compute("inference")
        self.log_metrics(metrics, "inference")
        self.metric_handler.reset("inference")

    def on_train_batch_end(self, *args, **kwargs):
        metrics = self.metric_handler.compute("train")
        self.log_metrics(metrics, "train")
        self.metric_handler.reset("train")

    def on_train_batch_start(self, *args, **kwargs):
        # print("On train batch start")
        pass
