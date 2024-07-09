from ...utils import from_datasets
from . import dataset as DNR_DATASETS


def DivideAndRemasterDataModule(cfg, test_only=False):
    dataset = {}

    for split in cfg.data.datasets:
        if test_only and split != "test":
            continue

        dataset_cls_name = cfg.data.datasets[split].cls
        dataset_cls = DNR_DATASETS.__dict__[dataset_cls_name]

        dataset[split] = dataset_cls(fs=cfg.fs, **cfg.data.datasets[split].kwargs)

    return from_datasets(
        train_dataset=dataset.get("train", None),
        val_dataset=dataset.get("val", None),
        test_dataset=dataset["test"],
        batch_size=cfg.data.datamodule.kwargs.batch_size_per_gpu,
        num_workers=cfg.data.datamodule.kwargs.num_workers,
    )
