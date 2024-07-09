class OptimizationHandler:
    def __init__(self, optimizer_cls, scheduler_cls, cfg):
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls

        self.cfg = cfg

    def configure_optimizers(self, model):
        config_dict = {}

        optimizer = self.optimizer_cls(
            model.parameters(), **self.cfg.optimizer.optimizer.kwargs
        )
        config_dict["optimizer"] = optimizer

        if self.scheduler_cls is None:
            return config_dict

        scheduler_dict = {}

        scheduler = self.scheduler_cls(optimizer, **self.cfg.optimizer.scheduler.kwargs)

        scheduler_dict["scheduler"] = scheduler

        if self.cfg.optimizer.scheduler.get("interval", None) is not None:
            scheduler_dict["interval"] = self.cfg.optimizer.scheduler.kwargs.interval

        if self.cfg.optimizer.scheduler.get("frequency", None) is not None:
            scheduler_dict["frequency"] = self.cfg.optimizer.scheduler.kwargs.frequency

        if self.cfg.optimizer.scheduler.get("monitor", None) is not None:
            scheduler_dict["monitor"] = self.cfg.optimizer.scheduler.kwargs.monitor

        if self.cfg.optimizer.scheduler.get("strict", None) is not None:
            scheduler_dict["strict"] = self.cfg.optimizer.scheduler.kwargs.strict

        if self.cfg.optimizer.scheduler.get("name", None) is not None:
            scheduler_dict["name"] = self.cfg.optimizer.scheduler.kwargs.delay

        config_dict["optimizer"] = optimizer
        config_dict["lr_scheduler"] = scheduler_dict

        return config_dict
