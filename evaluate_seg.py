import os
import cv2
import torch
import einops
import numpy as np
from pathlib import Path
from noise2same.backbone import swinia, unet, unet_b2u
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from noise2same import util
from noise2same.dataset.getter import expand_dataset_cfg
from importlib import reload
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple
from noise2same.util import crop_as
from noise2same.evaluator import Evaluator


from noise2same.backbone.swinir import SwinIR
from noise2same.backbone.unet import UNet
from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.tiling import TiledImageFactory
from noise2same.dataset.util import PadAndCropResizer
from noise2same.denoiser import Denoiser
from noise2same.util import crop_as, calculate_scores, detach_to_np


import logging
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_toolbelt.inference.tiles import TileMerger
from torch import Tensor as T
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import datetime
import glob
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
from typing import Dict, Optional, Tuple

import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm.auto import tqdm

from noise2same import util
from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.dataset.getter import (
    get_planaria_dataset_and_gt,
)
from noise2same.dataset.tiling import TiledImageFactory
from noise2same.evaluator import Evaluator
from noise2same.psf.psf_convolution import instantiate_psf

from sklearn.cluster import KMeans, DBSCAN
from skimage.filters import threshold_otsu


# from noise2same.backbone.swinir import SwinIR
# from noise2same.backbone.unet import UNet
# from noise2same.dataset.abc import AbstractNoiseDataset
# from noise2same.dataset.tiling import TiledImageFactory
# from noise2same.dataset.util import PadAndCropResizer
# from noise2same.denoiser import Denoiser
# from noise2same.util import crop_as, calculate_scores, detach_to_np

log = logging.getLogger(__name__)


from sklearn.metrics import f1_score, accuracy_score, precision_score


def accuracy_seg(pred_mask, gt_mask):
    return np.max([accuracy_score(gt_mask.flatten(), pred_mask.flatten()), accuracy_score(gt_mask.flatten(), 1 - pred_mask.flatten())])


def precision_seg(pred_mask, gt_mask):
    return np.max([precision_score(gt_mask.flatten(), pred_mask.flatten()), precision_score(gt_mask.flatten(), 1 - pred_mask.flatten())])


def get_seg_masks(batch): # ToDO: batch size is 1, extend to batch size > 1
    feature_map = batch['image'][0].transpose(1,2,0).reshape(-1, batch['shape'][0])
    k_means = KMeans(n_clusters=2)
    k_means.fit(feature_map)
    out = k_means.predict(feature_map)
    out = out.reshape(batch['shape'][1:])
    return out


def f1_score_seg(pred_mask, gt_mask):
    print(pred_mask.shape, gt_mask.shape)
    return np.max([f1_score(gt_mask.flatten(), pred_mask.flatten()), f1_score(gt_mask.flatten(), 1 - pred_mask.flatten())])


def iou(pred_mask, gt_mask):
    print(pred_mask.shape, gt_mask.shape)
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Input masks must have the same shape")

    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)

    if np.sum(union) == 0:
        iou_0 =  0.0
    else:
        iou_0 = np.sum(intersection) / np.sum(union)

    intersection = np.logical_and(gt_mask, 1 - pred_mask)
    union = np.logical_or(gt_mask, 1 - pred_mask)

    if np.sum(union) == 0:
        iou_1 = 0.0
    else:
        iou_1 = np.sum(intersection) / np.sum(union)

    return np.max([iou_0, iou_1])


def seg_scores(pred_mask, gt_mask, prefix='seg_'):
    metrics_fn = {'accuracy': accuracy_seg, 'precision': precision_seg, 'f1': f1_score_seg, 'iou': iou}
    scores = {prefix + metric: metrics_fn[metric](pred_mask, gt_mask) for metric in metrics_fn.keys()}
    return scores


class SegmentationEvaluator(Evaluator):
    def __init__(self,
            denoiser: Denoiser,
            device: str = "cuda",
            checkpoint_path: Optional[str] = None,):
        super().__init__(denoiser, device, checkpoint_path)

    @torch.no_grad()
    def evaluate(
            self,
            dataset: AbstractNoiseDataset,
            factory: Optional[TiledImageFactory] = None,
            half: bool = False,
            empty_cache: bool = False,
            key: str = 'image',
            keep_images: bool = False,
            min_max_scale: bool = False,
            metrics: Tuple[str, ...] = ("rmse", "psnr", "ssim"),
            num_workers: int = 0,
    ) -> List[Dict[str, Union[float, np.ndarray]]]:
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        self.model.eval()
        outputs = []
        iterator = tqdm(loader, desc="inference", position=0, leave=True)
        full_inference_time, test_size = 0, 0
        if factory is None:
            inference = partial(self._inference_batch, half=half, empty_cache=empty_cache)
        else:
            inference = partial(self._inference_large_batch, factory=factory, half=half,
                                empty_cache=empty_cache, keys=(key,))
        for i, batch in enumerate(iterator):
            out, inference_time = inference(batch)
            batch['input/image'] = batch.pop('image')  # rename to avoid key collision
            # TODO might not be optimal to copy the batch. Consider popping the keys
            out.update(batch)  # merge input batch and output
            # out = self._revert_batch(out, [key, 'ground_truth'])
            out['image'] = out['image'].detach().cpu().numpy()
            out['shape'] = out['shape'][0]
            out['image'] = crop_as(out['image'],
                                   (out['image'].shape[0], out['image'].shape[1], out['shape'][0], out['shape'][1]))
            out['shape'] = out['image'].shape
            pred_mask = get_seg_masks(out)
            out['pred_mask'] = pred_mask
            for j, (pred, gt) in enumerate(zip(out['pred_mask'], out['gt_mask'])):
                gt = gt.numpy().squeeze(2)
                scores = seg_scores(pred, gt)
                outputs.append(scores)
            full_inference_time += inference_time
            test_size += batch['ground_truth'].shape[0]

        log.info(f"Average inference time: {full_inference_time / test_size * 1000:.2f} ms")
        return outputs


def evaluate(
        evaluator: Evaluator,
        dataset: AbstractNoiseDataset,
        cfg: DictConfig,
        factory: Optional[TiledImageFactory] = None,
        train_dir: Optional[str] = None,
        save_results: bool = True,
        verbose: bool = True,
        keep_images: bool = False,
        metrics: Tuple[str, ...] = ("accuracy", "precision", "iou", "f1"),
):
    train_dir = train_dir or ''
    log.info(f"Evaluate dataset {str(dataset)} for key {cfg.evaluate.key} in train_dir {train_dir}")
    scores = evaluator.evaluate(dataset, factory,
                                num_workers=cfg.training.num_workers,
                                half=cfg.training.amp,
                                empty_cache=False,
                                key=cfg.evaluate.key,
                                keep_images=keep_images,
                                metrics=metrics,
                                min_max_scale=cfg.experiment in ('imagenet', 'hanzi'),
                                )
    # TODO do not create a list of None if there are no images
    predictions = {cfg.evaluate.key: [s.pop(cfg.evaluate.key, None) for s in scores]}
    scores = pd.DataFrame(scores)

    # Label each score with its dataset name and repeat id
    datasets = [dataset] if not isinstance(dataset, ConcatDataset) else dataset.datasets
    scores = scores.assign(
        dataset_name=np.concatenate([[str(ds)] * len(ds) for ds in datasets]),
        repeat_id=np.concatenate([np.arange(len(ds)) // (len(ds) // ds.n_repeats) for ds in datasets])
    )
    evaluation_dir = Path(train_dir) / f'evaluate' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if save_results:
        log.info(f"Saving results to {evaluation_dir / 'scores.csv'}")
        scores.to_csv(evaluation_dir / "scores.csv")
        if keep_images:
            np.savez(evaluation_dir / "predictions.npz", **predictions)

    if verbose and any(ds.n_repeats > 1 for ds in datasets):
        log.info("\nBefore averaging over repeats:\n" +
                 pformat(scores.groupby(["dataset_name", "repeat_id"]).mean()))
    scores = scores.groupby("dataset_name").mean().drop(columns="repeat_id")

    if verbose:
        log.info("\nEvaluation results:\n" + pformat(scores))

    scores = scores.to_dict()
    # Flatten scores dict as "metric.dataset" to make it compatible with wandb
    scores = {f"{metric}{'.'+dataset_name if len(datasets) > 1 else ''}": score
              for metric, dataset_dict in scores.items()
              for dataset_name, score in dataset_dict.items()}
    return scores


def main(exp_dir, config_path, checkpoint_path, other_args=None):
    cfg = OmegaConf.load(config_path)
    cfg.cwd = Path(os.getcwd())
    OmegaConf.resolve(cfg)

    expand_dataset_cfg(cfg)
    dataset = instantiate(cfg.dataset_test)
    factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None


    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    denoiser = instantiate(cfg.denoiser, backbone=backbone, head=head)

    # checkpoint = torch.load(checkpoint_path)['model']
    #
    # denoiser.load_state_dict(checkpoint)
    # denoiser.to('cuda')
    # denoiser.eval()

    evaluator = SegmentationEvaluator(denoiser, checkpoint_path=checkpoint_path)

    _ = evaluate(
        evaluator=evaluator,
        dataset=dataset,
        cfg=cfg,
        factory=factory,
        train_dir=exp_dir,
        verbose=True,
        keep_images=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", required=True,)
    parser.add_argument("--config_path", required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint_path",
                        help="The checkpoint to evaluate")
    args, unknown_args = parser.parse_known_args()

    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO,
                        filename=f'{args.exp_dir}/evaluate.log')
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    log.info('Start evaluation')

    util.register_config_resolvers()
    main(args.exp_dir, args.config_path, args.checkpoint_path, unknown_args)