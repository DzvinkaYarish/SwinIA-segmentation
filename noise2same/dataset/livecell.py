import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict
import cv2
from PIL import Image
import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import (
    add_microscope_blur_2d,
    add_noise,
    normalize,
)

from tqdm import tqdm

from pycocotools.coco import COCO

@dataclass
class LiveCellDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "/gpfs/space/projects/transformers_uss"
    mode: str = "train"
    add_blur_and_noise: bool = False

    def __str__(self) -> str:
        return f'livecell_{self.mode}_{self.part}{"_deconv" if self.add_blur_and_noise else ""}'

    def _validate(self) -> None:
        assert self.mode in ("train", "val", "test")

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        coco = COCO(os.path.join(self.path, f'livecell/annotations/livecell_coco_{self.mode}.json'))
        imgs_paths = coco.loadImgs(coco.getImgIds())

        folder = 'livecell_train_val_images' if self.mode in ('train', 'val') else 'livecell_test_images'
        images = np.concatenate([np.expand_dims(np.array(Image.open(os.path.join(self.path, 'livecell/images', folder, img['file_name']))), 0) for img in tqdm(imgs_paths)])
        seg_masks = np.concatenate([np.expand_dims(np.sum(np.array([coco.annToMask(obj) for obj in coco.loadAnns(coco.getAnnIds(imgIds=img['id']))]), axis=0), 0) for img in tqdm(imgs_paths)])
        seg_masks[seg_masks > 0] = 1

        return {
            'image': images,
            'ground_truth': images,
            'seg_mask': seg_masks
        }


    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image = self.image_index['image'][i]
        # TODO understand why is it here
        self.mean = np.mean(image, keepdims=True, dtype=np.float32)[None, ...]
        self.std = np.std(image, keepdims=True, dtype=np.float32)[None, ...]
        return {'image': image, 'ground_truth': self.image_index['ground_truth'][i], 'seg_mask': self.image_index['seg_mask'][i]}

    def _add_blur_and_noise(self, image: np.ndarray) -> np.ndarray:
        image = normalize(image)
        # TODO parametrize
        try:
            image, self.psf = add_microscope_blur_2d(image, size=17)
        except ValueError as e:
            raise ValueError(f"Failed to convolve image {image.shape}") from e
        image = add_noise(
                image,
                alpha=0.001,
                sigma=0.1,
                sap=0,  # 0.01 by default but it is not common to have salt and pepper
                quant_bits=10)
        return image * 255

