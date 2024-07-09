import math
from typing import List

import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F
from torchaudio.io import StreamReader
from tqdm import tqdm


class BaseChunkedInferenceHandler(nn.Module):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "reflect",
        rank: int = 0,
    ):
        super().__init__()

        self.fs = fs

        self.chunk_size_samples = int(chunk_size_seconds * fs)
        self.hop_size_samples = int(hop_size_seconds * fs)
        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples

        self.scaler = self.chunk_size_samples / (2 * self.hop_size_samples)

        window_fn = torch.__dict__[window_fn]

        if wkwargs is None:
            wkwargs = {}

        scaled_window = (
            window_fn(self.chunk_size_samples, **wkwargs)[None, None, :] / self.scaler
        )

        self.register_buffer("scaled_window", scaled_window)

        self.pad_mode = pad_mode
        self.inference_batch_size = inference_batch_size

        self.front_pad_samples = 2 * self.overlap_samples

        self.rank = rank

    def set_rank(self, rank: int):
        self.rank = rank

    def forward(self, mixture: torch.Tensor, model: nn.Module):
        raise NotImplementedError

    def _get_n_chunks_smart(self, n_samples: int):
        n_chunk_front_pad_only = (
            int(
                math.ceil(
                    (n_samples + self.front_pad_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )

        n_end_pad_samples = self._get_end_pad_samples(n_samples, n_chunk_front_pad_only)

        if n_end_pad_samples >= self.front_pad_samples:
            return n_chunk_front_pad_only

        else:
            n_chunk_front_pad_and_end_pad = (
                int(
                    math.ceil(
                        (
                            n_samples
                            + 2 * self.front_pad_samples
                            - self.chunk_size_samples
                        )
                        / self.hop_size_samples
                    )
                )
                + 1
            )

            return n_chunk_front_pad_and_end_pad

    def _get_n_chunks(self, n_samples: int):
        return (
            int(
                math.ceil(
                    (n_samples + 2 * self.front_pad_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )

    def _get_end_pad_samples(self, n_samples: int, n_chunks: int):
        return (
            (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples - n_samples
        )

    def _get_padded_samples(self, n_samples: int, n_chunks: int, end_pad_samples: int):
        return n_samples + 2 * self.front_pad_samples + end_pad_samples

    def _unfold(self, segment: torch.Tensor):
        batch_size, n_channels, _ = segment.shape

        assert batch_size == 1

        segment = segment.reshape(n_channels, 1, -1, 1)  # (n_channels, 1, n_samples, 1)

        unfolded_segment = F.unfold(
            segment,
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # (n_channels, chunk_size_samples, n_chunks)

        unfolded_segment = unfolded_segment.permute(
            0, 2, 1
        )  # (n_channels, n_chunks, chunk_size_samples)

        return unfolded_segment


class StandardTensorChunkedInferenceHandler(BaseChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "reflect",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            inference_batch_size=inference_batch_size,
            fs=fs,
            window_fn=window_fn,
            wkwargs=wkwargs,
            pad_mode=pad_mode,
            rank=rank,
        )

    def _fold(self, stem_output: torch.Tensor, n_samples: int, padded_samples: int):
        stem_output = stem_output * self.scaled_window.to(stem_output.device)
        stem_output = torch.permute(
            stem_output, (0, 2, 1)
        )  # (n_channels, chunk_size_samples, n_chunks)

        print(stem_output.shape)

        stem_output = F.fold(
            stem_output,
            output_size=(padded_samples, 1),
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # (n_channels, 1, padded_samples, 1)

        stem_output = stem_output[
            None,
            :,
            0,
            self.front_pad_samples : self.front_pad_samples + n_samples,
            0,
        ]

        return stem_output

    def _cat_and_fold(
        self, stem_outputs: List[torch.Tensor], n_samples: int, padded_samples: int
    ):
        stem_output = torch.cat(
            stem_outputs, dim=1
        )  # (n_channels, n_chunks, chunk_size_samples)

        stem_output = self._fold(stem_output, n_samples, padded_samples)

        return stem_output

    def _pad_and_unfold(self, mixture: torch.Tensor):
        batch_size, _, n_samples = mixture.shape

        assert batch_size == 1

        n_chunks = self._get_n_chunks(n_samples)
        end_pad_samples = self._get_end_pad_samples(n_samples, n_chunks)
        padded_samples = self._get_padded_samples(n_samples, n_chunks, end_pad_samples)

        if self.front_pad_samples >= n_samples:
            reflect_pad = (n_samples - 1, n_samples - 1)
            remaining_pad = self.front_pad_samples - (n_samples - 1)
            constant_pad = (remaining_pad, remaining_pad + end_pad_samples)
        elif self.front_pad_samples + end_pad_samples >= n_samples:
            reflect_pad = (self.front_pad_samples, n_samples - 1)
            remaining_pad = self.front_pad_samples + end_pad_samples - (n_samples - 1)
            constant_pad = (0, remaining_pad)
        else:
            reflect_pad = (
                self.front_pad_samples,
                self.front_pad_samples + end_pad_samples,
            )
            constant_pad = None

        padded_mixture = F.pad(mixture, reflect_pad, mode=self.pad_mode)

        if constant_pad is not None:
            padded_mixture = F.pad(
                padded_mixture,
                constant_pad,
                mode="constant",
            )

        unfolded_mixture = self._unfold(padded_mixture)

        # (n_chunks, n_channels, chunk_size_samples)

        return unfolded_mixture, n_samples, padded_samples

    def _tensor_forward(self, mixture: torch.Tensor, model: nn.Module):
        _, n_channels, n_samples = mixture.shape

        unfolded_mixture, n_samples, padded_samples = self._pad_and_unfold(mixture)
        # (n_channels, n_chunks, chunk_size_samples)

        # print(unfolded_mixture.shape)

        n_chunks = unfolded_mixture.shape[1]
        n_batches = math.ceil(n_chunks / self.inference_batch_size)
        outputs = {stem: [] for stem in model.stems}

        for i in tqdm(
            range(n_batches), position=self.rank + 1, desc=f"Rank {self.rank}"
        ):
            start = i * self.inference_batch_size
            end = min((i + 1) * self.inference_batch_size, n_chunks)
            chunk = unfolded_mixture[:, start:end, :]
            input_dict = {
                "mixture": {"audio": chunk.reshape(-1, 1, self.chunk_size_samples)}
            }
            output = model(input_dict)

            del chunk

            for stem in model.stems:
                outputs[stem].append(
                    output["estimates"][stem]["audio"].reshape(
                        n_channels, -1, self.chunk_size_samples
                    )
                )

            del output

        final_outputs = {
            stem: {
                "audio": self._cat_and_fold(outputs[stem], n_samples, padded_samples)
            }
            for stem in model.stems
        }

        return {"estimates": final_outputs}

    def forward(self, mixture: torch.Tensor, model: nn.Module):
        return self._tensor_forward(mixture, model)


class StandardFileChunkedInferenceHandler(StandardTensorChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "reflect",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            inference_batch_size=inference_batch_size,
            fs=fs,
            window_fn=window_fn,
            wkwargs=wkwargs,
            pad_mode=pad_mode,
            rank=rank,
        )

    def forward(self, mixture_path: str, model: nn.Module):
        audio, fs = ta.load(mixture_path)

        assert fs == self.fs

        mixture = audio[None, :, :]  # (1, n_channels, n_samples)

        return self._tensor_forward(mixture, model)


class BaseStreamingChunkedInferenceHandler(BaseChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "constant",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds,
            hop_size_seconds,
            inference_batch_size,
            fs,
            window_fn,
            wkwargs,
            pad_mode,
            rank,
        )

        assert (
            self.chunk_size_samples % self.hop_size_samples == 0
        ), "Chunk size must be a multiple of hop size"
        assert (
            self.overlap_samples % self.hop_size_samples == 0
        ), "Overlap must be a multiple of hop size"
        assert (
            self.front_pad_samples % self.hop_size_samples == 0
        ), "Front pad must be a multiple of hop size"

        self.streaming_frame_size_samples = self.hop_size_samples

        """
        
        □: a hop-size frame
        
        hops per chunk: 4
        
        chunks per batch: 5
        
        
        full audio:     □□□□□□□□□□□□□□□□□□□□□
        
        # batch 1:
        chunk 1:        0123
        chunk 2:         1234
        chunk 3:          2345
        chunk 4:           3456
        chunk 5:            4567
        buffer:         01234567
        
        # batch 2:
        chunk 6:             5678
        chunk 7:              6789
        chunk 8:               789a
        chunk 9:                89ab
        chunk 10:                9abc
        old buffer:     ■■■■■567
        new data:               89abc
        new buffer:          56789abc
        buffer idx:          01234567
        roll amount: 3
        
        """

        self.n_frames_per_chunk = self.chunk_size_samples // self.hop_size_samples
        self.n_frames_per_batch = (
            self.n_frames_per_chunk + self.inference_batch_size - 1
        )
        self.n_skip_frames_per_batch = self.inference_batch_size
        self.n_reusable_frames_per_batch = self.n_frames_per_chunk - 1
        self.n_new_frames_per_batch = self.inference_batch_size
        self.buffer_frame_start_idx = self.n_reusable_frames_per_batch

        self.n_frames_per_front_pad = self.front_pad_samples // self.hop_size_samples

        assert self.n_skip_frames_per_batch >= self.n_frames_per_front_pad

    def _batch_idx_to_chunk_idx(self, batch_idx: int):
        return batch_idx * self.inference_batch_size

    def _chunk_idx_to_start_sample_idx(self, chunk_idx: int):
        return chunk_idx * self.hop_size_samples

    def _chunk_idx_to_end_sample_idx(self, chunk_idx: int):
        return chunk_idx * self.hop_size_samples + self.chunk_size_samples

    def _batch_idx_to_start_sample_idx(self, batch_idx: int):
        return self._chunk_idx_to_start_sample_idx(
            self._batch_idx_to_chunk_idx(batch_idx)
        )

    def _batch_idx_to_end_sample_idx(self, batch_idx: int):
        return self._chunk_idx_to_start_sample_idx(
            self._batch_idx_to_chunk_idx(batch_idx) + self.inference_batch_size
        )

    def _get_mixture_shape(self, mixture):
        raise NotImplementedError

    def _grab_next_chunks(self, mixture, start_chunk_idx, n_samples):
        raise NotImplementedError

    def _forward(self, mixture, model):
        n_channels, n_samples = self._get_mixture_shape(mixture)

        n_chunks = self._get_n_chunks_smart(n_samples)
        end_pad_samples = self._get_end_pad_samples(n_samples, n_chunks)
        padded_samples = self._get_padded_samples(n_samples, n_chunks, end_pad_samples)

        n_batches = math.ceil(n_chunks / self.inference_batch_size)

        output_buffers = {
            stem: torch.zeros(1, n_channels, padded_samples, device=self.device)
            for stem in model.stems
        }

        input_buffer = torch.zeros(
            1,
            n_channels,
            self.n_frames_per_batch,
            self.hop_size_samples,
            device=self.device,
        )

        for batch_idx in range(n_batches):
            start_chunk_idx = self._batch_idx_to_chunk_idx(batch_idx)
            start_sample_idx = self._batch_idx_to_start_sample_idx(batch_idx)
            end_sample_idx = self._batch_idx_to_end_sample_idx(batch_idx)

            input_buffer = torch.roll(
                input_buffer, shifts=-self.n_skip_frames_per_batch, dims=2
            )

            input_buffer[:, :, self.buffer_frame_start_idx :, :] = (
                self._grab_next_chunks(mixture, start_chunk_idx, n_samples, n_channels)
            )

            input_buffer_reshaped = input_buffer.reshape(1, n_channels, -1)

            unfolded_chunk_mixture = self._unfold(input_buffer_reshaped)

            chunk_outputs = model({"mixture": {"audio": unfolded_chunk_mixture}})

            for stem in model.stems:
                folded_chunk_output = self._fold(
                    chunk_outputs[stem]["audio"], n_samples, padded_samples
                )
                output_buffers[stem][:, :, start_sample_idx:end_sample_idx] += (
                    folded_chunk_output
                )

        final_outputs = {
            stem: {
                "audio": output_buffers[stem][
                    self.front_pad_samples : self.front_pad_samples + n_samples
                ]
            }
            for stem in model.stems
        }

        return {"estimates": final_outputs}


class StreamingTensorChunkedInferenceHandler(BaseChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "constant",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds,
            hop_size_seconds,
            inference_batch_size,
            fs,
            window_fn,
            wkwargs,
            pad_mode,
            rank,
        )

    def _get_mixture_shape(self, mixture: torch.Tensor):
        batch_size, n_channels, n_samples = mixture.shape
        assert batch_size == 1
        return n_channels, n_samples

    def _grab_next_chunks(
        self,
        mixture: torch.Tensor,
        start_chunk_idx: int,
        n_samples: int,
        n_channels: int,
    ):
        unpadded_start_chunk_idx = start_chunk_idx - self.n_frames_per_front_pad

        start_sample_idx = self._chunk_idx_to_start_sample_idx(unpadded_start_chunk_idx)
        end_sample_idx = self._chunk_idx_to_end_sample_idx(
            start_chunk_idx + self.inference_batch_size
        )

        if start_sample_idx < 0:
            start_sample_idx = 0
            front_pad_samples = -start_sample_idx
        else:
            front_pad_samples = 0

        if end_sample_idx > n_samples:
            end_sample_idx = n_samples
            end_pad_samples = end_sample_idx - n_samples
        else:
            end_pad_samples = 0

        chunk = mixture[:, :, start_sample_idx:end_sample_idx]

        if front_pad_samples > 0 or end_pad_samples > 0:
            chunk = F.pad(
                chunk,
                (front_pad_samples, end_pad_samples),
                mode=self.pad_mode,
            )

        return chunk.reshape(1, n_channels, -1, self.hop_size_samples)

    def forward(self, mixture: torch.Tensor, model: nn.Module):
        return self._forward(mixture, model)


class StreamingFileChunkedInferenceHandler(BaseStreamingChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "constant",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            inference_batch_size=inference_batch_size,
            fs=fs,
            window_fn=window_fn,
            wkwargs=wkwargs,
            pad_mode=pad_mode,
            rank=rank,
        )

        self.current_chunk_idx = 0

    def _get_mixture_shape(self, streamer: StreamReader):
        stream_info = streamer.get_src_stream_info(streamer.default_audio_stream)

        n_channels = stream_info.num_channels
        n_samples = stream_info.num_frames

        return n_channels, n_samples

    def _grab_next_chunks(
        self,
        streamer: StreamReader,
        start_chunk_idx: int,
        n_samples: int,
        n_channels: int,
    ):
        unpadded_start_chunk_idx = start_chunk_idx - self.n_frames_per_front_pad

        n_chunks_to_grab = self.n_new_frames_per_batch

        if unpadded_start_chunk_idx < 0 and self.current_chunk_idx == 0:
            start_chunk_idx = 0
            n_chunks_to_grab -= self.n_frames_per_front_pad
            front_pad_chunks = self.n_frames_per_front_pad
            # only do this once, the next time we will rely on rolling the buffer
        else:
            front_pad_chunks = 0

        chunks = []

        for _ in range(n_chunks_to_grab):
            chunk = streamer.get_next_chunk()  # (n_samples, n_channels)
            self.current_chunk_idx += 1
            if chunk is None:
                break
            chunks.append(chunk.T)

        if len(chunks) < n_chunks_to_grab:
            end_pad_chunks = n_chunks_to_grab - len(chunks)
        else:
            end_pad_chunks = 0

        chunks = torch.stack(chunks, dim=1)[
            None, ...
        ]  # (1, n_channels, n_chunks, n_samples)

        if front_pad_chunks > 0 or end_pad_chunks > 0:
            chunks = F.pad(
                chunks,
                (0, 0, front_pad_chunks, end_pad_chunks),
                mode=self.pad_mode,
            )

        return chunks

    def forward(self, mixture_path: str, model: nn.Module):
        streamer = StreamReader(mixture_path)

        streamer.add_basic_audio_stream(
            frames_per_chunk=self.hop_size_samples,
            buffer_chunk_size=self.n_frames_per_batch,
            sample_rate=self.fs,
        )

        return self._forward(streamer, model)
