import pytorch_lightning as pl


class BaseEndToEndModule(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
