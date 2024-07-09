from torch import nn

from ..utils import deep_access_nested_dict

import torch


class MetricHandler(nn.Module):
    def __init__(self, metric_dict):
        super().__init__()

        self.metric_dict = nn.ModuleDict(
            {
                f"{split}_metrics": nn.ModuleDict(
                    {k: v["metric"] for k, v in split_dict.items()}
                )
                for split, split_dict in metric_dict.items()
            }
        )

        self.key_dict = {
            f"{split}_metrics": {k: v["keys"] for k, v in split_dict.items()}
            for split, split_dict in metric_dict.items()
        }
        
        self.updated = {
            mode: {
                k: False for k in self.metric_dict[mode]
            } for mode in self.metric_dict
        }

    def _update(self, batch, mode, return_metrics=False):
        # print(f"Retuning metrics: {return_metrics}")

        mode = f"{mode}_metrics"

        metrics = self.metric_dict[mode]

        for k, metric_module in metrics.items():
            key_pred = self.key_dict[mode][k]["pred"]
            key_true = self.key_dict[mode][k]["target"]
            y_pred = deep_access_nested_dict(batch, key_pred, strict=False)
            y_true = deep_access_nested_dict(batch, key_true, strict=False)

            metric_module.update(
                y_pred.detach(), y_true.detach(), return_metrics=True
            )  # return_metrics=return_metrics)
            
            self.updated[mode][k] = True

    def update(self, batch, mode):
        self._update(batch, mode, return_metrics=False)

    def compute(self, mode):
        mode = f"{mode}_metrics"
        metrics = self.metric_dict[mode]

        metric_dict = {}

        for k, metric_module in metrics.items():
            if not self.updated[mode][k]:
                metric_dict[k] = torch.tensor(torch.nan, device=metric_module.device)
            else:
                metric_dict[k] = metric_module.compute()

        return metric_dict

    def update_and_compute_batch(self, batch, mode):
        self._update(batch, mode, return_metrics=True)

        return self.compute_batch(batch, mode)

    def compute_batch(self, batch, mode):
        mode = f"{mode}_metrics"
        metrics = self.metric_dict[mode]

        metric_dict = {}

        for k, metric_module in metrics.items():
            
            if not self.updated[mode][k]:
                continue
            
            key_pred = self.key_dict[mode][k]["pred"]
            key_true = self.key_dict[mode][k]["target"]
            y_pred = deep_access_nested_dict(batch, key_pred).detach()
            y_true = deep_access_nested_dict(batch, key_true).detach()

            metric_dict[k] = metric_module.compute_batch(y_pred, y_true)

        return metric_dict

    def reset(self, mode):
        mode = f"{mode}_metrics"
        metrics = self.metric_dict[mode]

        for k, metric_module in metrics.items():
            if not self.updated[mode][k]:
                continue
            
            metric_module.reset()
            
            self.updated[mode][k] = False
