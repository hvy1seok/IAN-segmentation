import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import random
import time

import numpy as np
import torch
import torchio as tio
from hashlib import shake_256
from munch import munchify, unmunchify
from torch import nn
from os import path
import wandb
from tqdm import tqdm

from ian_segmentation.transforms.augmentations import AugFactory
from ian_segmentation.models import ModelFactory
from ian_segmentation.optimizers import OptimizerFactory
from ian_segmentation.schedulers import SchedulerFactory
from ian_segmentation.losses import LossFactory
from eval import Eval as Evaluator
from ian_segmentation.datasets.maxillo import Maxillo
from torch.utils.data import DataLoader


eps = 1e-10
class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json'
        if self.debug:
            filename = 'splits.json.small'

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)
        wandb.watch(self.model, log_freq=10)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
                sched_name,
                self.optimizer,
                milestones=sched_milestones,
                gamma=sched_gamma,
                mode='max',
                verbose=True,
                patience=15
            ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='train',
                transform=tio.Compose([
                    tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                    self.config.data_loader.preprocessing,
                    self.config.data_loader.augmentations,
                    ]),
                # dist_map=['sparse','dense']
        )
        self.val_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='val',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
        )
        self.test_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='test',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
        )
        self.synthetic_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='synthetic',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense'],
        ) 

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)
        self.synthetic_loader = self.synthetic_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.config.title,
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)

    def load(self):
        path = self.config.trainer.checkpoint
        logging.info(f'Loading checkpoint from {path}')
        state = torch.load(path)

        if 'title' in state.keys():
            # check that the title headers (without the hash) is the same
            self_title_header = self.config.title[:-11]
            load_title_header = state['title'][:-11]
            if self_title_header == load_title_header:
                self.config.title = state['title']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['state_dict'])
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def extract_data_from_patch(self, patch):
        volume = patch['data'][tio.DATA].float().cuda()
        gt = patch['dense'][tio.DATA].float().cuda()

        if 'Generation' in self.__class__.__name__:
            sparse = patch['sparse'][tio.DATA].float().cuda()
            images = torch.cat([volume, sparse], dim=1)
        else:
            images = volume

        emb_codes = torch.cat((
            patch[tio.LOCATION][:,:3],
            patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        return images, gt, emb_codes

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader
        if self.config.data_loader.training_set == 'generated':
            logging.info('using the generated dataset')
            data_loader = self.synthetic_loader

        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt, emb_codes = self.extract_data_from_patch(d)

            partition_weights = 1
            # TODO: Do only if not Competitor
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
            if torch.sum(gt_count) == 0: continue
            partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            preds = self.model(images, emb_codes)

            assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
            loss = self.loss(preds, gt, partition_weights)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            preds = (preds > 0.5).squeeze().detach()

            gt = gt.squeeze()
            self.evaluator.compute_metrics(preds, gt)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')

        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }

        wandb.log({
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Dice': epoch_dice,
            f'Train/IoU': epoch_iou,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        })

        return epoch_train_loss, epoch_iou


    def test(self, phase):

        self.model.eval()

        # with torch.no_grad():
        with torch.inference_mode():
            self.evaluator.reset_eval()
            losses = []

            if phase == 'Test':
                dataset = self.test_dataset
            elif phase == 'Validation':
                dataset = self.val_dataset

            for i, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):

                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        0
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler)
                gt_aggregator = tio.inference.GridAggregator(sampler)

                for j, patch in enumerate(loader):
                    images, gt, emb_codes = self.extract_data_from_patch(patch)

                    preds = self.model(images, emb_codes)
                    aggregator.add_batch(preds, patch[tio.LOCATION])
                    gt_aggregator.add_batch(gt, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                gt = gt_aggregator.get_output_tensor()
                partition_weights = 1

                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                if torch.sum(gt_count) != 0:
                    partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))

                loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
                losses.append(loss.item())

                output = output.squeeze(0)
                output = (output > 0.5)

                self.evaluator.compute_metrics(output, gt)

            epoch_loss = sum(losses) / len(losses)
            epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)

            wandb.log({
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss,
                f'{phase}/Dice': epoch_dice,
                f'{phase}/IoU': epoch_iou
            })

            return epoch_iou, epoch_dice


class Generation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)

    def inference(self, output_path):

        self.model.eval()

        with torch.no_grad():
            dataset = Maxillo(
                    self.config.data_loader.dataset,
                    'splits.json',
                    # splits=['train','val','test'],
                    splits='synthetic',  # issue 8: https://github.com/AImageLab-zip/alveolar_canal/issues/8
                    transform=self.config.data_loader.preprocessing,
                    dist_map=['sparse', 'dense']
            )
            crop_or_pad_transform = tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0)
            for i, subject in tqdm(enumerate(dataset), total=len(dataset)):
                directory = os.path.join(output_path, f'{subject.patient}')
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, 'generated.npy')

                if os.path.exists(file_path) and False:
                    logging.info(f'skipping {subject.patient}...')
                    continue

                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        patch_overlap=self.config.data_loader.grid_overlap,
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')

                logging.info(f'patient {subject.patient}...')
                for j, patch in enumerate(loader):
                    images = patch['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                    sparse = patch['sparse'][tio.DATA].float().cuda()
                    emb_codes = patch[tio.LOCATION].float().cuda()

                    # join sparse + data
                    images = torch.cat([images, sparse], dim=1)
                    output = self.model(images, emb_codes)  # BS, Classes, Z, H, W
                    aggregator.add_batch(output, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                # output = tio.CropOrPad(original_shape, padding_mode=0)(output)
                output = output.squeeze(0)
                # output = (output > 0.5).int()
                output = output.detach().cpu().numpy()  # BS, Z, H, W

                np.save(file_path, output)
                logging.info(f'patient {subject.patient} completed, {file_path}.')



class Segmentation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)


class ExperimentFactory:
    def __init__(self, config, debug=False):
        self.name = config.experiment.name
        self.config = config
        self.debug = debug

    def get(self):
        if self.name == 'Segmentation':
            experiment = Segmentation(self.config, self.debug)
        elif self.name == 'Generation':
            experiment = Generation(self.config, self.debug)
        else:
            raise ValueError(f'Experiment \'{self.name}\' not found')
        return experiment


# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5) # output len: 2*5=10
    return h.upper()

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="config.yaml", help="the config file to be used to run the experiment", required=True)
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    wandb.init(
        project="alveolar_canal_lee",
        entity="ian-segmentation",
        config=unmunchify(config)
    )

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preproc = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preproc = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preproc_file:
            preproc = yaml.load(preproc_file, yaml.FullLoader)
    config.data_loader.preprocessing = AugFactory(preproc).get_transform()

    # check if augmentations is set and file exists
    logging.info(f'loading augmentations')
    if config.data_loader.augmentations is None:
        aug = []
    elif not os.path.exists(config.data_loader.augmentations):
        logging.warning(f'Augmentations file does not exist: {config.augmentations}')
        aug = []
    else:
        with open(config.data_loader.augmentations) as aug_file:
            aug = yaml.load(aug_file, yaml.FullLoader)
    config.data_loader.augmentations = AugFactory(aug).get_transform()

    # make title unique to avoid overriding
    config.title = f'{config.title}_{timehash()}'

    logging.info(f'Instantiation of the experiment')
    # pdb.set_trace()
    experiment = ExperimentFactory(config, args.debug).get()
    logging.info(f'experiment title: {experiment.config.title}')

    project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.title)
    os.makedirs(project_dir_title, exist_ok=True)
    logging.info(f'project directory: {project_dir_title}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(project_dir_title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(experiment.config.data_loader.dataset):
        logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
        raise SystemExit

    # pre-calculate the checkpoints path
    checkpoints_path = path.join(project_dir_title, 'checkpoints')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
        raise SystemExit


    best_val = float('-inf')
    best_test = {
            'value': float('-inf'),
            'epoch': -1
            }


    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs+1):
            experiment.train()

            val_iou, val_dice = experiment.test(phase="Validation")
            logging.info(f'Epoch {epoch} Val IoU: {val_iou}')
            logging.info(f'Epoch {epoch} Val Dice: {val_dice}')

            if val_iou < 1e-05 and experiment.epoch > 15:
                logging.warning('WARNING: drop in performances detected.')

            optim_name = experiment.optimizer.name
            sched_name = experiment.scheduler.name

            if experiment.scheduler is not None:
                if optim_name == 'SGD' and sched_name == 'Plateau':
                    experiment.scheduler.step(val_iou)
                else:
                    experiment.scheduler.step(epoch)

            if epoch % 5 == 0:
                test_iou, test_dice = experiment.test(phase="Test")
                logging.info(f'Epoch {epoch} Test IoU: {test_iou}')
                logging.info(f'Epoch {epoch} Test Dice: {test_dice}')

                if test_iou > best_test['value']:
                    best_test['value'] = test_iou
                    best_test['epoch'] = epoch

            experiment.save('last.pth')

            if val_iou > best_val:
                best_val = val_iou
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best test IoU found: {best_test['value']} at epoch: {best_test['epoch']}
                ''')

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')

    # Do the inference
    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        experiment.inference(os.path.join(config.data_loader.dataset,'SPARSE'))
        # experiment.inference('/homes/llumetti/out')

# TODO: add a Final test metric
