from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio as ta
from torch.utils import data


class BaseSourceSeparationDataset(data.Dataset):
    def __init__(
        self,
        split: str,
        stems: List[str],
        files: List[str],
        data_path: str,
        fs: int,
        npy_memmap: bool = True,
        chunk_size_seconds: float | None = None,
        hop_size_seconds: float | None = None,
        deterministic_chunking: bool = False,
        aligned_chunking: bool = False,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = True,
        recompute_mixture: bool = False,
        mixture_stem_name: str = "mixture",
        target_dataset_length: int | None = None,
        stem_choices=None,
        random_stem_choices=False,
    ):
        self.split = split
        self.stems = stems
        self.files = files
        self.data_path = data_path
        self.fs = fs
        self.npy_memmap = npy_memmap

        self.chunked = chunk_size_seconds is not None
        self.deterministic_chunking = deterministic_chunking if self.chunked else None
        self.chunk_size_seconds = chunk_size_seconds
        self.hop_size_seconds = hop_size_seconds

        self.chunk_size_samples = (
            int(chunk_size_seconds * fs) if chunk_size_seconds is not None else None
        )
        self.hop_size_samples = (
            int(hop_size_seconds * fs) if hop_size_seconds is not None else None
        )

        self.aligned_chunking = aligned_chunking

        if self.deterministic_chunking:
            assert self.hop_size_seconds is not None
            assert self.aligned_chunking == True

        self.full_audio_length_seconds = full_audio_length_seconds
        self.full_audio_length_samples = (
            int(full_audio_length_seconds * fs)
            if full_audio_length_seconds is not None
            else None
        )

        self.auto_include_mixture = auto_include_mixture
        self.recompute_mixture = recompute_mixture
        self.mixture_stem_name = mixture_stem_name

        if self.auto_include_mixture:
            if mixture_stem_name not in self.stems:
                self.stems = [self.mixture_stem_name] + self.stems

        self.stem_no_mixture = [s for s in self.stems if s != self.mixture_stem_name]

        self.identifiers = self._make_index()

        self.true_dataset_length = len(self.identifiers)

        if target_dataset_length is not None:
            self.target_dataset_length = target_dataset_length
        else:
            self.target_dataset_length = self.true_dataset_length

        self.stem_choices = stem_choices
        self.random_stem_choices = random_stem_choices

    def _make_index(self):
        if self.deterministic_chunking:
            assert self.hop_size_seconds is not None
            assert self.full_audio_length_samples is not None

            n_chunks = (
                self.full_audio_length_samples - self.chunk_size_samples
            ) // self.hop_size_samples

            identifiers = []

            for file in self.files:
                for chunk_index in range(n_chunks):
                    identifier = {"file": file, "chunk_index": chunk_index}

                    identifiers.append(identifier)
        else:
            identifiers = [{"file": file} for file in self.files]

        return identifiers

    def __getitem__(self, index: int):
        identifier = self.get_identifier(index)

        if self.stem_choices is not None:
            if self.random_stem_choices:
                stems_ = [
                    np.random.choice(stem_choices) for stem_choices in self.stem_choices
                ]
                # print(f"Random stems: {stems_}")
            else:
                stems_ = [
                    stem_choices[index % len(stem_choices)]
                    for stem_choices in self.stem_choices
                ]
        else:
            if self.auto_include_mixture:
                if self.recompute_mixture:
                    stems_ = self.stem_no_mixture
                else:
                    stems_ = self.stems
            else:
                stems_ = self.stem_no_mixture

        audio = self.get_audio(identifier, stems=stems_)

        if self.mixture_stem_name in audio or self.auto_include_mixture:
            if self.recompute_mixture:
                if self.mixture_stem_name in audio:
                    audio.pop(self.mixture_stem_name)
                mixture = sum(audio.values())
            else:
                mixture = audio.pop(self.mixture_stem_name)
        else:
            mixture = None

        nan_ = torch.full_like(list(audio.values())[0], torch.nan)

        out = {
            "sources": {
                stem: {"audio": audio.get(stem, nan_)} for stem in self.stem_no_mixture
            },
            "identifier": identifier,
        }

        if mixture is not None:
            out["mixture"] = {"audio": mixture}

        return out

    def __len__(self):
        return self.target_dataset_length

    def get_stem_path(self, *, stem: str, identifier: Dict[str, Any]) -> str:
        raise NotImplementedError

    def load_stem(self, stem_path: str) -> torch.Tensor | np.memmap:
        if self.npy_memmap:
            # do not cast to torch.Tensor, as it will load the whole file into memory
            return np.load(stem_path, mmap_mode="r")
        else:
            audio, fs = ta.load(stem_path)
            assert fs == self.fs

            return audio

    def chunk_stem(
        self, stem_audio: torch.Tensor | np.memmap, start_index: int | None
    ) -> torch.Tensor | np.memmap:
        if start_index is None:
            if self.full_audio_length_samples is None:
                audio_length_samples = stem_audio.shape[-1]
            else:
                audio_length_samples = self.full_audio_length_samples
                # assert stem_audio.shape[-1] == audio_length_samples, (stem_audio.shape[-1], audio_length_samples)

            start_index = np.random.randint(
                0, audio_length_samples - self.chunk_size_samples
            )

        stem_audio = stem_audio[:, start_index : start_index + self.chunk_size_samples]

        assert stem_audio.shape[-1] == self.chunk_size_samples

        return stem_audio

    def chunk_audio(
        self, audio: Dict[str, torch.Tensor | np.memmap]
    ) -> Dict[str, torch.Tensor | np.memmap]:
        if self.aligned_chunking:
            audio_length_samples_dict = {k: v.shape[-1] for k, v in audio.items()}
            min_audio_length_samples = min(audio_length_samples_dict.values())

            start_index = np.random.randint(
                0, min_audio_length_samples - self.chunk_size_samples
            )

            chunked_audio = {
                k: self.chunk_stem(v, start_index=start_index) for k, v in audio.items()
            }
        else:
            chunked_audio = {
                k: self.chunk_stem(v, start_index=None) for k, v in audio.items()
            }

        return chunked_audio

    def get_stem(
        self,
        *,
        stem: str,
        identifier: Dict[str, Any],
        chunk_now: bool = True,
        chunk_params: Dict[str, Any],
    ) -> torch.Tensor:
        stem_path = self.get_stem_path(stem=stem, identifier=identifier)
        stem_audio = self.load_stem(stem_path)

        chunked_stem_audio = (
            self.chunk_stem(stem_audio, **chunk_params) if chunk_now else stem_audio
        )

        return chunked_stem_audio

    def get_audio(
        self, identifier: Dict[str, Any], stems: List[str] = None
    ) -> Dict[str, torch.Tensor | np.memmap]:
        if self.chunked:
            if self.deterministic_chunking:
                chunk_now = True
                chunk_index = identifier["chunk_index"]
                chunk_params = {
                    "start_index": chunk_index * self.hop_size_samples,
                }
            elif self.aligned_chunking:
                chunk_now = True
                start_index = np.random.randint(
                    0, self.full_audio_length_samples - self.chunk_size_samples
                )

                chunk_params = {"start_index": start_index}
            else:
                chunk_now = True
                chunk_params = {"start_index": None}
        else:
            chunk_now = False
            chunk_params = {}

        audio = {}

        for stem in stems:
            audio[stem] = self.get_stem(
                stem=stem,
                identifier=identifier,
                chunk_now=chunk_now,
                chunk_params=chunk_params,
            )

        if self.chunked and not chunk_now:
            audio = self.chunk_audio(audio)

        if self.npy_memmap:
            audio = {
                k: torch.from_numpy(v.astype(np.float32)) for k, v in audio.items()
            }

        return audio

    def get_identifier(self, index: int) -> Dict[str, Any]:
        identifier = self.identifiers[index % self.true_dataset_length]

        return identifier
