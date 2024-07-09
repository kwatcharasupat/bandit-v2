import torch
from torch.nn.modules.loss import _Loss


class LpSNR(_Loss):
    def __init__(
        self,
        p: int = 2,
        multiplier: float = None,
        epsilon: float = 1.0e-3,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        if multiplier is None:
            multiplier = 10.0 * (2.0 / p)

        self.p = p
        self.multiplier = multiplier
        self.epsilon = epsilon

    def compute_snr(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.flatten(y_pred, start_dim=1)
        y_true = torch.flatten(y_true, start_dim=1)
        error = y_pred - y_true

        target_energy = torch.mean(torch.pow(torch.abs(y_true), self.p), dim=-1)
        error_energy = torch.mean(torch.pow(torch.abs(error), self.p), dim=-1)

        snr = self.multiplier * torch.log10(
            (target_energy + self.epsilon) / (error_energy + self.epsilon)
        )

        return -snr.mean()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(y_pred):
            y_pred = torch.view_as_real(y_pred)
            y_true = torch.view_as_real(y_true)

            snr_real = self.compute_snr(y_pred[..., 0], y_true[..., 0])
            snr_imag = self.compute_snr(y_pred[..., 1], y_true[..., 1])

            snr = snr_real + snr_imag

        else:
            snr = self.compute_snr(y_pred, y_true)

        return -snr.mean()


class L1SNR(LpSNR):
    def __init__(
        self, multiplier: float = None, epsilon: float = 1e-3, reduction: str = "mean"
    ):
        super().__init__(
            p=1, multiplier=multiplier, epsilon=epsilon, reduction=reduction
        )

    # def compute_snr(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    #     y_pred = torch.flatten(y_pred, start_dim=1)
    #     y_true = torch.flatten(y_true, start_dim=1)
    #     error = y_pred - y_true

    #     target_energy = torch.mean(torch.abs(y_true), dim=-1)
    #     error_energy = torch.mean(torch.abs(error), dim=-1)

    #     snr = self.multiplier * torch.log10((target_energy + self.epsilon) / (error_energy + self.epsilon))

    #     return -snr.mean()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(y_pred):
            return self._complex_forward(y_pred, y_true)
        else:
            return self._real_forward(y_pred, y_true)

    def _real_forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.flatten(y_pred, start_dim=1)
        y_true = torch.flatten(y_true, start_dim=1)
        
        nan_filter = torch.all(torch.isnan(y_true), dim=-1)
        
        if torch.all(nan_filter):
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        y_pred = y_pred[~nan_filter]
        y_true = y_true[~nan_filter]
        
        error = y_pred - y_true

        target_energy = torch.nanmean(torch.abs(y_true), dim=-1)
        error_energy = torch.nanmean(torch.abs(error), dim=-1)

        snr = self.multiplier * torch.log10(
            (target_energy + self.epsilon) / (error_energy + self.epsilon)
        )

        snr = torch.mean(snr, dim=-1)

        return - torch.nanmean(snr)

    def _complex_forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        y_pred = torch.view_as_real(y_pred)
        y_true = torch.view_as_real(y_true)

        snr_real = self._real_forward(y_pred[..., 0], y_true[..., 0])
        snr_imag = self._real_forward(y_pred[..., 1], y_true[..., 1])

        return snr_real + snr_imag
