import torch
from torch import nn

from ..utils import deep_access_nested_dict

from collections import defaultdict

from itertools import combinations


class LossHandler(nn.Module):
    def __init__(self, loss_dict, optional_keys=None, use_combination=False):
        super().__init__()

        self.loss_dict = nn.ModuleDict({k: v["loss"] for k, v in loss_dict.items()})

        for k, v in loss_dict.items():
            self.register_buffer(f"{k}_weight", torch.tensor(v["weight"]))

        self.key_dict = {k: v["keys"] for k, v in loss_dict.items()}

        self.optional_keys = optional_keys if optional_keys is not None else []

        self.use_combination = use_combination

        if self.use_combination:
            print("Using combination loss")

            losses_by_domain = defaultdict(list)

            for k in self.loss_dict.keys():
                domain = k.split("/")[1]

                losses_by_domain[domain].append(k)

            loss_combinations = {}

            for domain in losses_by_domain.keys():
                combis = combinations(losses_by_domain[domain], 2)

                for l1, l2 in combis:
                    if l1.split("/")[0] == l2.split("/")[0]:
                        
                        if l1.split("/")[-1].split("_")[0] != l2.split("/")[-1].split("_")[0]:
                            
                            k = l1 + "+" + l2.split("/")[-1]
                            
                            loss_combinations[k] = (l1, l2)

            self.loss_combinations = loss_combinations

            print(
                f"The following {len(self.loss_combinations)} combinations will be used:"
            )
            for k, (l1, l2) in self.loss_combinations.items():
                print(k)

    def forward(self, batch):
        if self.use_combination:
            return self._combination_forward(batch)
        else:
            loss_dict, _, _ = self._single_forward(batch)
            return loss_dict

    def _single_forward(self, batch):
        loss_dict = {"total": 0.0, "components": {}}

        y_preds_ = {}
        y_trues_ = {}

        for k, loss_module in self.loss_dict.items():
            key_pred = self.key_dict[k]["pred"]
            key_true = self.key_dict[k]["target"]
            y_pred_ = deep_access_nested_dict(batch, key_pred)
            y_true_ = deep_access_nested_dict(batch, key_true)

            y_preds_[k] = y_pred_
            y_trues_[k] = y_true_

            batch_filter = torch.isnan(y_true_)
            if torch.all(batch_filter):
                print(f"All values are NaN for {k}")
                loss_dict["components"][k] = torch.tensor(
                    0.0, device=y_true_.device, requires_grad=True
                )
            else:
                loss_dict["components"][k] = loss_module(y_pred_, y_true_)

            loss_dict["total"] += (
                self.__getattr__(f"{k}_weight") * loss_dict["components"][k]
            )

        return loss_dict, y_preds_, y_trues_

    def _combination_forward(self, batch):
        loss_dict, y_preds_, y_trues_ = self._single_forward(batch)

        for k, (l1, l2) in self.loss_combinations.items():
            y_pred_ = y_preds_[l1] + y_preds_[l2]
            y_true_ = y_trues_[l1] + y_trues_[l2]

            loss_module = self.loss_dict[l1]

            batch_filter = torch.isnan(y_true_)

            if torch.all(batch_filter):
                print(f"All values are NaN for {k}")
                loss_dict["components"][k] = torch.tensor(
                    0.0, device=y_true_.device, requires_grad=True
                )
            else:
                loss_dict["components"][k] = loss_module(y_pred_, y_true_)

            weight = self.__getattr__(f"{l1}_weight") + self.__getattr__(f"{l2}_weight")

            loss_dict["total"] += 0.5 * weight * loss_dict["components"][k]

        return loss_dict
