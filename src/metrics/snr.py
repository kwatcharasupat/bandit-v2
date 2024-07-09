from typing import Any, Tuple

import torch
import torchmetrics as tm
from torchmetrics.audio.snr import SignalNoiseRatio
from torchmetrics.functional.audio.snr import (
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)


def safe_signal_noise_ratio(
    preds: torch.Tensor, target: torch.Tensor, zero_mean: bool = False
) -> torch.Tensor:
    return torch.nan_to_num(
        signal_noise_ratio(preds, target, zero_mean=zero_mean),
        nan=torch.nan,
        posinf=100.0,
        neginf=-100.0,
    )


def safe_scale_invariant_signal_noise_ratio(
    preds: torch.Tensor, target: torch.Tensor, zero_mean: bool = False
) -> torch.Tensor:
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape
    """
    return torch.nan_to_num(
        scale_invariant_signal_noise_ratio(preds, target),
        nan=torch.nan,
        posinf=100.0,
        neginf=-100.0,
    )


def decibels(x: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    mean_squared = torch.mean(torch.square(x), dim=-1)
    n_samples = x.shape[0]
    return torch.sum(10 * torch.log10(mean_squared + threshold)), n_samples


class Decibels(tm.Metric):
    def __init__(self, threshold: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

        self.add_state("running_mean", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("running_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y):
        db, count = decibels(y, self.threshold)
        self.running_mean += db.cpu()
        self.running_count += count

    def compute(self) -> torch.Tensor:
        return self.running_mean / self.running_count

    # def reset(self) -> None:
    #     self.running_mean = torch.tensor(0.0)
    #     self.running_count = torch.tensor(0)


class PredictedDecibels(Decibels):
    def update(self, ypred, ytrue) -> None:
        return super().update(ypred)


class TargetDecibels(Decibels):
    def update(self, ypred, ytrue) -> None:
        return super().update(ytrue)


class SafeSignalNoiseRatio(SignalNoiseRatio):
    def __init__(
        self,
        zero_mean: bool = False,
        threshold: float = 1e-6,
        fs: int = 44100,
        **kwargs: Any,
    ) -> None:
        super().__init__(zero_mean, **kwargs)

        self.threshold = threshold
        self.fs = fs

        self.sample_mismatch_thresh_seconds = 0.1

        self.add_state("snr_list", default=[], dist_reduce_fx="cat")

    def _fix_shape(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples_preds = preds.shape[-1]
        n_samples_target = target.shape[-1]

        if n_samples_preds != n_samples_target:
            if (
                abs(n_samples_preds - n_samples_target) / self.fs
                > self.sample_mismatch_thresh_seconds
            ):
                raise ValueError(
                    "The difference between the number of samples of the predictions and the target is too large (100 ms)"
                )

            min_samples = min(n_samples_preds, n_samples_target)
            preds = preds[..., :min_samples]
            target = target[..., :min_samples]

        return preds, target

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, return_metrics=True
    ) -> None:
        """Update state with predictions and targets."""

        snr_batch = self.compute_batch(preds, target)

        self.snr_list.append(snr_batch)

        return snr_batch.clone()

    def compute(self) -> torch.Tensor:
        """Compute metric."""

        if len(self.snr_list) == 0:
            return torch.tensor(float("nan"))

        if isinstance(self.snr_list, list):
            snr_list = torch.cat(self.snr_list, dim=0)
        elif isinstance(self.snr_list, torch.Tensor):
            snr_list = self.snr_list
        else:
            raise ValueError("snr_list must be a list or a tensor")

        return torch.nanmedian(snr_list)

    def compute_batch(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds, target = self._fix_shape(preds, target)

        snr_batch = safe_signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )

        return snr_batch


SafeSignalNoiseRatioMedian = SafeSignalNoiseRatio


class SafeSignalNoiseRatioMean(SignalNoiseRatio):
    def __init__(
        self,
        zero_mean: bool = False,
        threshold: float = 1e-6,
        fs: int = 44100,
        **kwargs: Any,
    ) -> None:
        super().__init__(zero_mean, **kwargs)

        self.threshold = threshold
        self.fs = fs

        self.sample_mismatch_thresh_seconds = 0.1

        self.add_state("snr_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def _fix_shape(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples_preds = preds.shape[-1]
        n_samples_target = target.shape[-1]

        if n_samples_preds != n_samples_target:
            if (
                abs(n_samples_preds - n_samples_target) / self.fs
                > self.sample_mismatch_thresh_seconds
            ):
                raise ValueError(
                    "The difference between the number of samples of the predictions and the target is too large (100 ms)"
                )

            min_samples = min(n_samples_preds, n_samples_target)
            preds = preds[..., :min_samples]
            target = target[..., :min_samples]

        return preds, target

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""

        preds, target = self._fix_shape(preds, target)

        snr_batch = safe_signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )

        self.snr_total += snr_batch.sum()
        self.n_samples += snr_batch.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute metric."""

        if self.n_samples == 0:
            return torch.tensor(float("nan"))

        return self.snr_total / self.n_samples


class SafeScaleInvariantSignalNoiseRatio(SafeSignalNoiseRatio):
    def __init__(
        self,
        zero_mean: bool = False,
        threshold: float = 1e-6,
        fs: int = 44100,
        **kwargs: Any,
    ) -> None:
        super().__init__(zero_mean, threshold, fs, **kwargs)

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, return_metrics=True
    ) -> None:
        """Update state with predictions and targets."""

        snr_batch = self.compute_batch(preds, target)

        self.snr_list.append(snr_batch)

        # print(f"snr_batch: {snr_batch}")
        return snr_batch.clone()

    def compute_batch(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds, target = self._fix_shape(preds, target)

        snr_batch = safe_scale_invariant_signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )

        return snr_batch

    def compute(self) -> torch.Tensor:
        """Compute metric."""

        if len(self.snr_list) == 0:
            return torch.tensor(float("nan"))

        if isinstance(self.snr_list, list):
            snr_list = torch.cat(self.snr_list, dim=0)
        elif isinstance(self.snr_list, torch.Tensor):
            snr_list = self.snr_list
        else:
            raise ValueError("snr_list must be a list or a tensor")

        return torch.nanmedian(snr_list)


SafeScaleInvariantSignalNoiseRatioMedian = SafeScaleInvariantSignalNoiseRatio


class SafeScaleInvariantSignalNoiseRatioMean(SafeSignalNoiseRatioMean):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""

        preds, target = self._fix_shape(preds, target)

        snr_batch = safe_scale_invariant_signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )

        self.snr_total += snr_batch.sum()
        self.n_samples += snr_batch.shape[0]
