import os
from pathlib import Path
import numpy as np
import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from evaluate import evaluate
from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.denoiser import Denoiser
from noise2same.evaluator import Evaluator
from torch.utils.data import Subset


class SubsetAttr(Subset):
    """
    Wrapper for Subset that allows to access attributes of the wrapped dataset.
    """

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __str__(self) -> str:
        return str(self.dataset)


@pytest.fixture(scope="module", autouse=True)
def set_cwd():
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}


@pytest.mark.parametrize('dataset_name',
                         ['bsd68', 'hanzi', 'imagenet', 'fmd', 'fmd_deconvolution', 'synthetic',
                          'synthetic_grayscale', 'ssi', 'hela_shallow'])
def test_regular_dataset_inference(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    evaluator = Evaluator(Denoiser(), device='cpu')
    batch = next(iter(loader))
    out, _ = evaluator._inference_batch(batch)
    batch['input/image'] = batch.pop('image')  # rename to avoid key collision
    out.update(batch)  # merge input batch and output
    out = evaluator._revert_batch(out, ['image', 'input/image'])
    for shape, pred, orig in zip(batch['shape'], out['image'], out['input/image']):
        assert np.all(np.array(shape) == np.array(pred.shape))
        assert np.allclose(pred, orig)


@pytest.mark.parametrize('dataset_name', ['imagenet', 'planaria', 'hela'])
def test_tiled_dataset_inference(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    factory = instantiate(cfg.factory_test)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    evaluator = Evaluator(Denoiser(), device='cpu')
    batch = next(iter(loader))
    out, _ = evaluator._inference_large_batch(batch, factory)
    batch['input/image'] = batch.pop('image')  # rename to avoid key collision
    out.update(batch)  # merge input batch and output
    out = evaluator._revert_batch(out, ['image', 'input/image'])
    for shape, pred, orig in zip(batch['shape'], out['image'], out['input/image']):
        assert np.all(np.array(shape) == np.array(pred.shape))
        assert np.allclose(pred, orig, atol=1e-3)


@pytest.mark.parametrize('dataset_name', ['imagenet', 'planaria', 'hela'])
def test_tiled_dataset_evaluation(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    dataset = SubsetAttr(dataset, range(min(2, len(dataset))))
    factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None
    evaluator = Evaluator(Denoiser(), device='cpu')
    outputs = evaluator.evaluate(dataset, factory, metrics=('rmse',))
    assert outputs is not None


@pytest.mark.parametrize('dataset_name', ['bsd68', 'synthetic', 'imagenet'])
def test_full_evaluation(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+training.amp=True',
                     '+training.num_workers=8', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    if not dataset_name == 'synthetic':
        dataset = SubsetAttr(dataset, range(min(2, len(dataset))))
    factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None
    evaluator = Evaluator(Denoiser(), device='cpu')
    outputs = evaluate(evaluator, dataset, cfg, factory)
    assert outputs is not None
    for k, v in outputs.items():
        assert v > 0, f"Metric {k} is negative {v}"
