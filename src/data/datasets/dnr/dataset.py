import os
from typing import Any, Dict, List

from ...base import BaseSourceSeparationDataset


class BaseDivideAndRemasterDataset(BaseSourceSeparationDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        stems: List[str] = None,
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
    ):
        print("Initializing BaseDivideAndRemasterDataset")

        if stems is None:
            stems = ["speech", "music", "sfx"]

        subfolder = "npy32" if npy_memmap else "audio"
        data_path = os.path.join(data_root, subfolder, subset, split)

        print("Data Path: ", data_path)
        print("Loading files from: ", os.path.join(data_root, subfolder, subset, split))

        files = os.listdir(data_path)

        print("Files: ", len(files))

        super().__init__(
            split=split,
            stems=stems,
            files=files,
            data_path=data_path,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            deterministic_chunking=deterministic_chunking,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
        )

    def get_stem_path(self, *, stem: str, identifier: Dict[str, Any]) -> str:
        file = identifier["file"]

        if self.npy_memmap:
            return os.path.join(self.data_path, file, f"{stem}.npy")

        return os.path.join(self.data_path, file, f"{stem}.wav")


class DivideAndRemasterFullTrackDataset(BaseDivideAndRemasterDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        stems: List[str] = None,
        npy_memmap: bool = True,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = True,
        recompute_mixture: bool = False,
        mixture_stem_name: str = "mixture",
        target_dataset_length: int | None = None,
    ):
        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=None,
            hop_size_seconds=None,
            deterministic_chunking=None,
            aligned_chunking=None,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
        )


class DivideAndRemasterRandomChunkDataset(BaseDivideAndRemasterDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        chunk_size_seconds: float,
        target_dataset_length: int,
        stems: List[str] = None,
        npy_memmap: bool = True,
        aligned_chunking: bool = False,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = None,
        recompute_mixture: bool = None,
        mixture_stem_name: str = "mixture",
    ):
        if auto_include_mixture is None:
            auto_include_mixture = not aligned_chunking

        if recompute_mixture is None:
            recompute_mixture = not aligned_chunking

        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=None,
            deterministic_chunking=False,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
        )


class DivideAndRemasterDeterministicChunkDataset(BaseDivideAndRemasterDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        stems: List[str] = None,
        npy_memmap: bool = True,
        aligned_chunking: bool = True,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = None,
        recompute_mixture: bool = None,
        mixture_stem_name: str = "mixture",
        target_dataset_length: int = None,
    ):
        if auto_include_mixture is None:
            auto_include_mixture = not aligned_chunking

        if recompute_mixture is None:
            recompute_mixture = not aligned_chunking

        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            deterministic_chunking=True,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
        )


class BaseDivideAndRemasterRandomSingingDataset(BaseSourceSeparationDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        stems: List[str] = None,
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
        stem_choices = None,
        random_stem_choices = False
    ):
        print("Initializing BaseDivideAndRemasterRandomSingingDataset")

        if stems is None:
            stems = ["speech", "music", "music_w_singing", "sfx"]
            
        if stem_choices is None:
            stem_choices = [
                ["speech"],
                ["music", "music_w_singing"],
                ["sfx"],
            ]

        subfolder = "npy32" if npy_memmap else "audio"
        data_path = os.path.join(data_root, subfolder, subset, split)

        print("Data Path: ", data_path)
        print("Loading files from: ", os.path.join(data_root, subfolder, subset, split))

        files = os.listdir(data_path)

        print("Files: ", len(files))

        super().__init__(
            split=split,
            stems=stems,
            files=files,
            data_path=data_path,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            deterministic_chunking=deterministic_chunking,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
            stem_choices=stem_choices,
            random_stem_choices=random_stem_choices
        )

    def get_stem_path(self, *, stem: str, identifier: Dict[str, Any]) -> str:
        file = identifier["file"]

        if self.npy_memmap:
            return os.path.join(self.data_path, file, f"{stem}.npy")

        return os.path.join(self.data_path, file, f"{stem}.wav")
    
    
class DivideAndRemasterRandomSingingFullTrackDataset(BaseDivideAndRemasterRandomSingingDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        stems: List[str] = None,
        npy_memmap: bool = True,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = True,
        recompute_mixture: bool = False,
        mixture_stem_name: str = "mixture",
        target_dataset_length: int | None = None,
        stem_choices = None,
        random_stem_choices = False
    ):
        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=None,
            hop_size_seconds=None,
            deterministic_chunking=None,
            aligned_chunking=None,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
            stem_choices=stem_choices,
            random_stem_choices=random_stem_choices
        )
        
        
class DivideAndRemasterRandomSingingRandomChunkDataset(BaseDivideAndRemasterRandomSingingDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        chunk_size_seconds: float,
        target_dataset_length: int,
        stems: List[str] = None,
        npy_memmap: bool = True,
        aligned_chunking: bool = False,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = None,
        recompute_mixture: bool = None,
        mixture_stem_name: str = "mixture",
        stem_choices = None,
        random_stem_choices = True
    ):
        if auto_include_mixture is None:
            auto_include_mixture = not aligned_chunking

        if recompute_mixture is None:
            recompute_mixture = not aligned_chunking

        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=None,
            deterministic_chunking=False,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
            stem_choices=stem_choices,
            random_stem_choices=random_stem_choices
        )
        
class DivideAndRemasterRandomSingingDeterministicChunkDataset(BaseDivideAndRemasterRandomSingingDataset):
    def __init__(
        self,
        split: str,
        subset: str,
        data_root: str,
        fs: int,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        stems: List[str] = None,
        npy_memmap: bool = True,
        aligned_chunking: bool = True,
        full_audio_length_seconds: float | None = 60.0,
        auto_include_mixture: bool = None,
        recompute_mixture: bool = None,
        mixture_stem_name: str = "mixture",
        target_dataset_length: int = None,
        stem_choices = None,
        random_stem_choices = True
    ):
        if auto_include_mixture is None:
            auto_include_mixture = not aligned_chunking

        if recompute_mixture is None:
            recompute_mixture = not aligned_chunking

        super().__init__(
            split=split,
            subset=subset,
            stems=stems,
            data_root=data_root,
            fs=fs,
            npy_memmap=npy_memmap,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            deterministic_chunking=True,
            aligned_chunking=aligned_chunking,
            full_audio_length_seconds=full_audio_length_seconds,
            auto_include_mixture=auto_include_mixture,
            recompute_mixture=recompute_mixture,
            mixture_stem_name=mixture_stem_name,
            target_dataset_length=target_dataset_length,
            stem_choices=stem_choices,
            random_stem_choices=random_stem_choices
        )