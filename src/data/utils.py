import inspect
from typing import Any, Mapping, Optional, Sequence, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset


def from_datasets(
    train_dataset: Optional[
        Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]
    ] = None,
    val_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    test_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    predict_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    **datamodule_kwargs: Any,
) -> LightningDataModule:
    def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
        shuffle &= not isinstance(ds, IterableDataset)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def train_dataloader() -> TRAIN_DATALOADERS:
        assert train_dataset

        if isinstance(train_dataset, Mapping):
            return {
                key: dataloader(ds, shuffle=True) for key, ds in train_dataset.items()
            }
        if isinstance(train_dataset, Sequence):
            return [dataloader(ds, shuffle=True) for ds in train_dataset]
        return dataloader(train_dataset, shuffle=True)

    def val_dataloader() -> EVAL_DATALOADERS:
        assert val_dataset

        if isinstance(val_dataset, Sequence):
            return [dataloader(ds) for ds in val_dataset]
        return dataloader(val_dataset)

    def test_dataloader() -> EVAL_DATALOADERS:
        assert test_dataset

        if isinstance(test_dataset, Sequence):
            return [dataloader(ds) for ds in test_dataset]
        return dataloader(test_dataset)

    def predict_dataloader() -> EVAL_DATALOADERS:
        assert predict_dataset

        if isinstance(predict_dataset, Sequence):
            return [dataloader(ds) for ds in predict_dataset]
        return dataloader(predict_dataset)

    def setup(stage):
        print("Setting up datamodule", stage)
        return

    candidate_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    accepted_params = inspect.signature(LightningDataModule.__init__).parameters
    accepts_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in accepted_params.values()
    )
    if accepts_kwargs:
        special_kwargs = candidate_kwargs
    else:
        accepted_param_names = set(accepted_params)
        accepted_param_names.discard("self")
        special_kwargs = {
            k: v for k, v in candidate_kwargs.items() if k in accepted_param_names
        }

    datamodule = LightningDataModule(**datamodule_kwargs, **special_kwargs)
    if train_dataset is not None:
        datamodule.train_dataloader = train_dataloader  # type: ignore[method-assign]
    if val_dataset is not None:
        datamodule.val_dataloader = val_dataloader  # type: ignore[method-assign]
    if test_dataset is not None:
        datamodule.test_dataloader = test_dataloader  # type: ignore[method-assign]
    if predict_dataset is not None:
        datamodule.predict_dataloader = predict_dataloader  # type: ignore[method-assign]

    datamodule.setup = setup

    return datamodule
