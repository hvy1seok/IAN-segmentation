import os
import random
import shutil
from typing import Sequence
import numpy as np

import torch
import torchio as tio
import yaml
from munch import munchify
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from mayavi import mlab

from ian_segmentation.datasets.maxillo import Maxillo
from ian_segmentation.models import ModelFactory
from ian_segmentation.transforms.augmentations import AugFactory
from ian_segmentation.utils import setup


def visualize_data(patient: int = 98, save_dir: os.PathLike = 'examples', splits: str = 'val') -> None:
    colormaps = ['Greys', 'Greens', 'Oranges']

    input_path = os.path.join(save_dir, splits, 'input', f'P{patient}.npy')
    gt_path = os.path.join(save_dir, splits, 'gt', f'P{patient}.npy')
    output_path = os.path.join(save_dir, splits, 'output', f'P{patient}.npy')

    data = [
        np.load(input_path),
        np.load(gt_path),
        np.load(output_path),
    ]

    src = mlab.pipeline.scalar_field(data[0], colormap='Blues')
    cut_plane = mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes')
    mlab.view(azimuth=270, elevation=270)

    for i, d in enumerate(data):
        print(d.shape)
        x, y, z = np.mgrid[:d.shape[0], :d.shape[1], :d.shape[2]]

        alpha = 0.1 if i == 0 else 0.9
        mlab.contour3d(x, y, z, d, colormap=colormaps[i], contours=8, opacity=alpha)
    
    mlab.show()


def save_predictions(config_path: os.PathLike, save_dir: os.PathLike, split: str = 'val', n_images: int = 10):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f'Removing old `{save_dir}` directory...')
    
    save_input_dir = os.path.join(save_dir, split, 'input')
    os.makedirs(save_input_dir)
    save_output_dir = os.path.join(save_dir, split, 'output')
    os.makedirs(save_output_dir)
    save_gt_dir = os.path.join(save_dir, split, 'gt')
    os.makedirs(save_gt_dir)
    print(f'New `{save_dir}` directory is generated')
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    config = munchify(config)

    setup(config.seed)

    with open(config.data_loader.preprocessing, 'r') as f:
        preproc = yaml.load(f, yaml.FullLoader)
    preproc_transforms = AugFactory(preproc).get_transform()

    filename = 'splits.json'

    num_classes = len(config.data_loader.labels)
    if 'Jaccard' in config.loss.name or num_classes == 2:
        num_classes = 1

    model_name = config.model.name
    in_ch = 1
    emb_shape = [dim // 8 for dim in config.data_loader.patch_shape]

    model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
    model = nn.DataParallel(model)

    val_dataset = Maxillo(
        root=config.data_loader.dataset,
        filename=filename,
        splits=split,
        transform=preproc_transforms,
    )
    
    path = config.trainer.checkpoint
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])

    model.eval()
    with torch.inference_mode():
        for subject in tqdm(val_dataset):
            sampler = tio.inference.GridSampler(
                subject,
                config.data_loader.patch_shape,
                0
            )
            loader = DataLoader(sampler, batch_size=config.data_loader.batch_size)
            image_aggregator = tio.inference.GridAggregator(sampler)
            aggregator = tio.inference.GridAggregator(sampler)
            gt_aggregator = tio.inference.GridAggregator(sampler)

            for patch in loader:
                images = patch['data'][tio.DATA].float().cuda()
                print('images.shape:', images.shape)
                gt = patch['dense'][tio.DATA].float().cuda()

                emb_codes = torch.cat((
                    patch[tio.LOCATION][:,:3],
                    patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
                ), dim=1).float().cuda()

                preds = model(images, emb_codes)
                print('preds.shape:', preds.shape)
                image_aggregator.add_batch(images, patch[tio.LOCATION])
                aggregator.add_batch(preds, patch[tio.LOCATION])
                gt_aggregator.add_batch(gt, patch[tio.LOCATION])

            image = image_aggregator.get_output_tensor()
            print('image.shape:', image.shape)
            image = image.squeeze(0)
            image = image.detach().cpu().numpy()  # BS, Z, H, W
            np.save(os.path.join(save_input_dir, f'{subject.patient}.npy'), image)
            print(f'The input of patient {subject.patient} is saved at {save_input_dir}')

            output = aggregator.get_output_tensor()
            print('output.shape:', output.shape)
            output = output.squeeze(0)
            # output = output > 0.5
            output = output.detach().cpu().numpy()  # BS, Z, H, W
            np.save(os.path.join(save_output_dir, f'{subject.patient}.npy'), output)
            print(f'The output of patient {subject.patient} is saved at {save_output_dir}')

            gt = gt_aggregator.get_output_tensor()
            print('gt.shape:', gt.shape)
            gt = gt.squeeze(0)
            gt = gt.detach().cpu().numpy()  # BS, Z, H, W
            np.save(os.path.join(save_gt_dir, f'{subject.patient}.npy'), gt)
            print(f'The gt of patient {subject.patient} is saved at {save_gt_dir}')
