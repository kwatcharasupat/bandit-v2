from types import SimpleNamespace

import pytorch_lightning as pl
import torchmetrics as tm
from omegaconf import DictConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torch import nn, optim
from torch.optim import lr_scheduler

from ..data.datasets.dnr.datamodule import DivideAndRemasterDataModule
from ..losses import l1snr
from ..metrics import snr
from ..models.bandit.bandit import Bandit
from .loss_handler import LossHandler
from .metric_handler import MetricHandler
from .optim_handler import OptimizationHandler
from .inference_handler import StandardTensorChunkedInferenceHandler
from .system import System

from datetime import datetime

METRIC_SEARCH_SPACE = [tm, snr]

OPTIMIZER_SEARCH_SPACE = [optim]

LR_SCHEDULER_SEARCH_SPACE = [lr_scheduler]

LOSS_SEARCH_SPACE = [nn, l1snr]

ALLOWED_DATAMODULES = [DivideAndRemasterDataModule]
ALLOWED_DATAMODULES_DICT = {dm.__name__: dm for dm in ALLOWED_DATAMODULES}

ALLOWED_ARCHITECTURES = [Bandit]
ALLOWED_ARCHITECTURES_DICT = {arch.__name__: arch for arch in ALLOWED_ARCHITECTURES}

LOGGER_SEARCH_SPACE = [pl.loggers]

CALLBACK_SEARCH_SPACE = [
    SimpleNamespace(RayTrainReportCallback=RayTrainReportCallback),
    pl.callbacks,
]

PLUGIN_SEARCH_SPACE = [
    SimpleNamespace(RayLightningEnvironment=RayLightningEnvironment),
    pl.plugins,
]

ALLOWED_INFERENCE_HANDLERS = [StandardTensorChunkedInferenceHandler]
ALLOWED_INFERENCE_HANDLERS_DICT = {ih.__name__: ih for ih in ALLOWED_INFERENCE_HANDLERS}


def build_logger(cfg: DictConfig):
    loggers = []

    for logger_cfg in cfg.trainer.loggers:
        found = False
        for lsp in LOGGER_SEARCH_SPACE:
            if logger_cfg.cls in lsp.__dict__:
                version = cfg.get("version", None)
                if version is None:
                    version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                logger = lsp.__dict__[logger_cfg.cls](
                    **logger_cfg.kwargs, version=version
                )
                loggers.append(logger)
                found = True
                break
        if not found:
            raise ValueError(f"Logger {logger_cfg.cls} not recognized")

    print(loggers)

    return loggers


def build_callbacks(cfg: DictConfig):
    callbacks = []

    for callback_cfg in cfg.trainer.callbacks:
        found = False
        for csp in CALLBACK_SEARCH_SPACE:
            if callback_cfg.cls in csp.__dict__:
                callback = csp.__dict__[callback_cfg.cls](**callback_cfg.kwargs)
                callbacks.append(callback)
                found = True
                break
        if not found:
            raise ValueError(f"Callback {callback_cfg.cls} not recognized")

    return callbacks


def build_strategy(cfg: DictConfig):
    if cfg.trainer.strategy is None:
        return "auto"

    if cfg.trainer.strategy.cls == "ddp":
        return "ddp"

    if cfg.trainer.strategy.cls == "RayDDPStrategy":
        return RayDDPStrategy()

    raise ValueError(f"Strategy {cfg.strategy.cls} not recognized")


def build_plugins(cfg: DictConfig):
    plugins = []

    for plugin_cfg in cfg.trainer.plugins:
        found = False
        for psp in PLUGIN_SEARCH_SPACE:
            if plugin_cfg.cls in psp.__dict__:
                plugin = psp.__dict__[plugin_cfg.cls](**plugin_cfg.kwargs)
                plugins.append(plugin)
                found = True
                break
        if not found:
            raise ValueError(f"Plugin {plugin_cfg.cls} not recognized")

    return plugins


def build_trainer(cfg: DictConfig, world_rank=0, use_ray=True):
    if not use_ray or world_rank == 0:
        callbacks = build_callbacks(cfg)
        logger = build_logger(cfg)
    else:
        callbacks = []
        logger = []

    if use_ray:
        callbacks.append(RayTrainReportCallback())

    strategy = build_strategy(cfg)
    plugins = build_plugins(cfg)

    trainer_kwargs = cfg.trainer.kwargs

    trainer = pl.Trainer(
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        **trainer_kwargs,
    )

    if use_ray:
        trainer = prepare_trainer(trainer)

    return trainer


def build_model(cfg: DictConfig):
    arch_name = cfg.models.cls
    arch_cls = ALLOWED_ARCHITECTURES_DICT[arch_name]

    if "Banquet" not in arch_name:
        kwargs = {**cfg.models.kwargs, "stems": cfg.data.commons.datasets.stems}

    else:
        kwargs = {**cfg.models.kwargs}

    model = arch_cls(fs=cfg.fs, **kwargs)

    return model


def build_loss_handler(cfg: DictConfig):
    loss_dict = {}

    optional_keys = []

    for loss in cfg.losses:
        found = False
        for lsp in LOSS_SEARCH_SPACE:
            if loss.cls in lsp.__dict__:
                loss_dict[loss.name] = {
                    "loss": lsp.__dict__[loss.cls](**loss.kwargs),
                    "keys": loss.keys_,
                    "weight": loss.weight,
                }

                if loss.get("optional", False):
                    optional_keys.append(loss.name)

                found = True
                break
        if not found:
            raise ValueError(f"Loss {loss.cls} not recognized")

    loss_handler = LossHandler(
        loss_dict,
        optional_keys=optional_keys,
        use_combination=cfg.get("use_combination_loss", False),
    )

    return loss_handler


def build_metric_handler(cfg: DictConfig):
    metric_dict = {}

    for split in cfg.metrics:
        metric_dict[split] = {}
        for metric in cfg.metrics[split]:
            found = False
            for msp in METRIC_SEARCH_SPACE:
                if metric.cls in msp.__dict__:
                    kwargs = metric.get("kwargs", {})
                    metric_dict[split][metric.name] = {
                        "metric": msp.__dict__[metric.cls](**kwargs),
                        "keys": metric.keys_,
                    }
                    found = True
                    break
            if not found:
                raise ValueError(f"Metric {metric.cls} not recognized")

    metric_handler = MetricHandler(metric_dict)

    return metric_handler


def build_optim_handler(cfg: DictConfig):
    optimizer_cls_name = cfg.optimizer.optimizer.cls

    found = False
    for osp in OPTIMIZER_SEARCH_SPACE:
        if optimizer_cls_name in osp.__dict__:
            optimizer_cls = osp.__dict__[optimizer_cls_name]
            found = True
            break
    if not found:
        raise ValueError(f"Optimizer {optimizer_cls_name} not recognized")

    if cfg.optimizer.get("scheduler", None) is None:
        scheduler_cls = None
    else:
        scheduler_cls_name = cfg.optimizer.scheduler.cls
        found = False
        for ssp in LR_SCHEDULER_SEARCH_SPACE:
            if scheduler_cls_name in ssp.__dict__:
                scheduler_cls = ssp.__dict__[scheduler_cls_name]
                found = True
                break
        if not found:
            raise ValueError(f"Scheduler {scheduler_cls_name} not recognized")

    optim_handler = OptimizationHandler(optimizer_cls, scheduler_cls, cfg)

    return optim_handler


def build_datamodule(cfg: DictConfig, test_only=False):
    dm_cls_name = cfg.data.datamodule.cls
    dm_cls = ALLOWED_DATAMODULES_DICT[dm_cls_name]
    dm = dm_cls(cfg, test_only=test_only)

    return dm


def build_inference(cfg: DictConfig):
    if "inference" not in cfg:
        return None

    inference_cls_name = cfg.inference.cls

    if inference_cls_name is None:
        return None

    inference_cls = ALLOWED_INFERENCE_HANDLERS_DICT[inference_cls_name]
    kwargs = cfg.inference.get("kwargs", {})
    inference_handler = inference_cls(fs=cfg.fs, **kwargs)

    return inference_handler


def build_system(cfg: DictConfig):
    model = build_model(cfg)
    loss_handler = build_loss_handler(cfg)
    metric_handler = build_metric_handler(cfg)
    optim_handler = build_optim_handler(cfg)
    inference_handler = build_inference(cfg)

    system = System(
        model=model,
        loss_handler=loss_handler,
        metric_handler=metric_handler,
        optim_handler=optim_handler,
        inference_handler=inference_handler,
        cfg=cfg,
    )

    return system
