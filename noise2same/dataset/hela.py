from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import h5py
import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.util import normalize_percentile

def read_h5py(path):
    with h5py.File(path, "r") as f:
        img = np.array(f["image"], dtype=np.float32)
    return img


@dataclass
class HelaShallowDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/hela"
    channel_id: int = 3
    mode: str = "train"
    standardize: bool = False

    def __str__(self) -> str:
        return f'hela_shallow_{self.mode}_ch{self.channel_id}'

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        input_files = sorted(list(Path(self.path).glob(f"reconvolved/*ch{self.channel_id}.h5")))
        gt_files = sorted(list(Path(self.path).glob(f"deconvolved/*ch{self.channel_id}.h5")))
        assert len(input_files) > 0, f"no files found in {self.path}"
        assert len(input_files) == len(gt_files), f"input and gt files have different lengths"
        if self.mode == "train":
            input_files = input_files[:-1]
            gt_files = gt_files[:-1]
        elif self.mode in ("val", "test"):
            input_files = input_files[-1:]
            gt_files = gt_files[-1:]
        else:
            raise ValueError(f"unknown mode {self.mode}")

        return {'image': np.stack([normalize_percentile(read_h5py(f), 0.1, 99.9) for f in input_files]),
                'ground_truth': np.stack([normalize_percentile(read_h5py(f), 0.1, 99.9) for f in gt_files])}

    def get_number_of_images(self) -> int:
        return self.image_index['image'].shape[0] * self.image_index['image'].shape[1]

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image_id = i // self.image_index['image'].shape[1]
        plane_id = i % self.image_index['image'].shape[1]
        return {k: v[image_id, plane_id] for k, v in self.image_index.items()}


@dataclass
class HelaDataset(HelaShallowDataset):
    n_dim = 3

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        image_index = super()._create_image_index()

        # TODO replace with PadIfNeeded for 3D data
        image_index = {k: np.pad(v, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant')
                       for k, v in image_index.items()}
        return image_index

    def __str__(self) -> str:
        return f'hela_{self.mode}_ch{self.channel_id}'

    def get_number_of_images(self) -> int:
        return self.image_index['image'].shape[0]

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
