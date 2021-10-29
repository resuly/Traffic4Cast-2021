#!/usr/bin/env python
# coding: utf-8

# pip install einops GPUtil tensorboard argparse pandas h5py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=2) # Batch Size
parser.add_argument('--epochs', default=5)
args = parser.parse_args()

import tqdm
import logging, os, json, glob, pickle, collections
import numpy as np
import pandas as pd
import random, h5py
import gc

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

# Data Loader
class T4CDatasetInandOut(Dataset):
    def __init__(
        self,
        root_dir: str = "../data/raw",
        files_path: list = [],
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False, # a little bit faster, but this format would create very large files 
        use_npz: bool = False, # run p0.RawData.py first, don't use this method, won't speed up too much.
        inandout: dict = None,
    ):
        """torch dataset from training data.

        Parameters
        ----------
        root_dir
            "./data/raw/" or an absolute dir
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.root_dir = root_dir
        self.limit = limit
        self.files = files_path
        self.use_npy = use_npy
        self.use_npz = use_npz
        self.inandout = inandout
        self.transform = transform
        self._load_dataset()
        
        # temp array in memory
        self.temp_name = None
        self.temp_array = None

    def _load_dataset(self):
        # self.files = list(Path(self.root_dir).rglob(self.file_filter))
        pass

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npz:            
            if self.temp_name is None or self.temp_name != fn:
                print(fn)
                self.temp_array = np.load(fn)['data']
                self.temp_name = fn
            data = self.temp_array[sl]
            return data
        elif self.use_npy:
            return np.load(fn)
        else:
            if self.temp_name is None or self.temp_name != fn:
                with h5py.File(str(fn) if isinstance(fn, Path) else fn, "r") as fr:
                    print(fn)
                    data = fr.get("array")
                    self.temp_array = np.array(data)
                    self.temp_name = fn
                    
                    torch.cuda.empty_cache()
                    gc.collect()
            return self.temp_array[sl] 

    def __len__(self):
        self.n_samples_per_day = 288 - self.inandout['input_len'] - self.inandout['output_len'] + 1
        n_samples = len(self.files) * self.n_samples_per_day
        if self.limit is not None:
            return min(n_samples, self.limit)
        return n_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")
        
        # start = time.time()
        
        file_idx = idx // self.n_samples_per_day
        sample_idx = idx % self.n_samples_per_day
        
        complete_sample = self._load_h5_file(self.files[file_idx], sl=slice(sample_idx, sample_idx + self.inandout['input_len'] + self.inandout['output_len']))
        input_data  = complete_sample[self.inandout['input_index']]
        output_data = complete_sample[self.inandout['output_index']]

        if self.transform is not None:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data
def local_sampler_288(dataset, inandout):
    n_samples = 288 - inandout['input_len'] - inandout['output_len'] + 1
    # n_samples is the number of sample per file
    n = len(dataset)
    raw_indices = np.array(range(n))
    ndays = len(dataset) // n_samples
    rest_days = n - ndays * n_samples
    dataset_indices = []
    for n in range(ndays):
        a = n * n_samples
        b = (n+1) * n_samples
        arr = raw_indices[a:b]
        np.random.shuffle(arr)
        dataset_indices.append(arr)
    if rest_days > 0:
        arr = raw_indices[ndays * n_samples:]
        dataset_indices.append(arr)
    return np.concatenate(dataset_indices)   
def create_train_file_df():
    all_train_files = list(Path('../data/').rglob(f"**/training/*8ch.h5"))
    arr = []
    for file in all_train_files:
        file_info = {}
        file_info['file'] = str(file)
        date, city = file.parts[-1].split('_')[:2]
        file_info['date'] = date
        file_info['city'] = city
        arr.append(file_info)
    df = pd.DataFrame(arr)
    df.date = pd.to_datetime(df.date)
    df.loc[:, 'dayofweek'] = df.date.dt.dayofweek
    df.loc[:, 'year'] = df.date.dt.year
    df.loc[:, 'month'] = df.date.dt.month
    df.loc[:, 'yeartype'] = df.city + df.year.astype(str)
    return df
def get_train_file(config):
    diff_dofw = None
    while diff_dofw != 7:
        df_temp = pd.concat([df[df.yeartype == k].sample(n=config[k])for k in df.yeartype.unique()])
        diff_dofw = len(df_temp.dayofweek.unique())
    df.drop(df[df.file.isin(df_temp.file.values)].index, inplace=True)
    files_train = [Path(f) for f in df_temp.file.values]
    random.shuffle(files_train)
    return files_train

# CONFIGS
file_config = {
    "MOSCOW2019": 1,
    "MOSCOW2020": 1,
    "ANTWERP2020": 1,
    "ANTWERP2019": 1,
    "BERLIN2019": 2,
    "CHICAGO2019": 2,
    "MELBOURNE2019": 3,
    "BARCELONA2019": 1,
    "BARCELONA2020": 1,
    "BANGKOK2020": 2,
    "BANGKOK2019": 2,
    "ISTANBUL2019": 2,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    'batch_size': int(args.bs),
    'num_epochs': int(args.epochs),
    'save_summary_steps': 10,
    "lr": 3e-4,
    "wd": 2e-5,
    'inandout': {
        'input_len': 12, 
        'output_len': 12, 
        'input_index': np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 
        'output_index': np.array([12, 13, 14, 17, 20, 23])
    }, 
    'city':'ALL',
    'file_num': sum([v for k, v in file_config.items()])
}
inlen = len(config['inandout']['input_index'])
outlen = len(config['inandout']['output_index'])
inoutmode = f'{inlen}to{outlen}'

# Model
class SparseUNet(ME.MinkowskiNetwork):
    def __init__(self, hs_block1=12, hs_block2=12, hs_block3=16, block3_tr=8, block2_tr=48):
        in_nchannel, out_nchannel, D = len(config['inandout']['input_index']), len(config['inandout']['output_index']), 3
        super(SparseUNet, self).__init__(D)
        self.block1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_nchannel,
                out_channels=hs_block1,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(hs_block1),
        )
        self.block2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=hs_block1,
                out_channels=hs_block2,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(hs_block2),
        )
        self.block3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=hs_block2,
                out_channels=hs_block3,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(hs_block3),
        )
        self.block3_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=hs_block3,
                out_channels=block3_tr,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(block3_tr),
        )
        self.block2_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=hs_block2+block3_tr,
                out_channels=block2_tr,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(block2_tr),
        )
        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=hs_block1+block2_tr,
            out_channels=out_nchannel,
            kernel_size=1,
            stride=1,
            dimension=D,
            expand_coordinates=True
        )

    def forward(self, x):
        # x to sparse tensor
        x = ME.MinkowskiOps.to_sparse(x)

        out_s1 = self.block1(x)
        out = MF.relu(out_s1)

        out_s2 = self.block2(out)
        out = MF.relu(out_s2)

        out_s4 = self.block3(out)
        out = MF.relu(out_s4)

        out = MF.relu(self.block3_tr(out))
        out = ME.cat(out, out_s2)
        out = MF.relu(self.block2_tr(out))
        out = ME.cat(out, out_s1)

        out = self.conv1_tr(out)

        dense_output, min_coord, tensor_stride = out.dense()
        missed_min_coordinate = np.array(dense_output.size()[-3:]) - np.array([495, 436, 8])
        if missed_min_coordinate.sum() != 0:
            dense_output, min_coord, tensor_stride = out.dense(
            min_coordinate=torch.IntTensor(missed_min_coordinate)
        )
        return dense_output

# Metrics
def rmse(predictions, targets):
    """Compute root mean squared error"""
    return torch.sqrt(((predictions - targets) ** 2).mean())
def mse(predictions, targets):
    """Compute mean squared error"""
    return ((predictions - targets) ** 2).mean()
def mae(predictions, targets):
    """Compute mean absolute error"""
    return torch.absolute((predictions - targets)).mean()
def get_next_version(root_dir):
    # get next version number 
    # Tensor Board root_dir/version_{}
    if not os.path.exists(root_dir):
        return 0
    
    existing_versions = []
    for d in os.listdir(root_dir):
        if d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))
            
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1

class Runner():
    def __init__(self, model_name, config=None):
        self.genertae_train_files()
        self.config = config
        self.model_name = model_name
        self.log_dir = f'./logs/{model_name}_ALL'
        self.model = globals()[model_name]()

        # DataParallel is not working with ME, need to use DDP, todo
        # if torch.cuda.device_count() > 1:
            # self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        self.scaler = amp.GradScaler(enabled=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.6, min_lr=1e-5, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)
        self.loss_fn = {'mse':nn.MSELoss(), 'kl': nn.KLDivLoss(),'kl-sum': nn.KLDivLoss(reduction='sum'), 'kl-none': nn.KLDivLoss(reduction='none'), 'ce': nn.CrossEntropyLoss(), 'L1':nn.L1Loss()}
        self.traffic_metrics = {'mse': mse,'rmse': rmse,'mae': mae}
        self.step = 0
        self.previous_epochs = 0
    def genertae_train_files(self):
        # generate training files
        df = []
        while len(df) == 0:
            df = create_train_file_df()
        resource_avail = True
        all_files = []
        while resource_avail:
            condition_w = set(range(7))
            yeartypes = list(file_config.keys())
            random.shuffle(yeartypes)
            df_temp_list = []
            for yeartype in yeartypes:
                if len(condition_w) == 0:
                    condition_w = set(range(7))
                w_i = np.random.randint(0, len(condition_w))
                w = list(condition_w)[w_i]

                df_t = df[(df.yeartype == yeartype)&(df.dayofweek == w)][:file_config[yeartype]]
                if len(df_t) == 0:
                    # print('no resources!')
                    resource_avail = False
                    break
                else:
                    df_temp_list.append(df_t)
                    condition_w = condition_w.difference(set([w]))
            df_temp = pd.concat(df_temp_list)
            df.drop(df[df.file.isin(df_temp.file.values)].index, inplace=True)
            files = [Path(f) for f in df_temp.file.values]
            random.shuffle(files)
            all_files += files
        self.files_train = all_files
        print(len(self.files_train))
    def train(self):
        # set model to training mode
        self.model.train()
        file_num = self.config['file_num']
        n = (self.current_epoch-1) % (len(self.files_train) // file_num)
        train_dataset = T4CDatasetInandOut(files_path=self.files_train[n*file_num:(n+1)*file_num], inandout=config['inandout'])
        dataloader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=False, sampler=local_sampler_288(train_dataset, config['inandout']))

        # summary for current training loop and a running average object for loss
        summ = []
        loop = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ascii=True, leave=False)
        loop_postfix = {}
        for i, (x, y) in loop:

            x, y = x.float().to(device), y.float().to(device)
            
            y_pred = self.model(x)

            loss_mse = self.loss_fn['mse'](y, y_pred)
            loss = loss_mse
            # loss = loss_mse

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update the loop
            loop.set_description(f"Epoch [{self.current_epoch}/{self.previous_epochs+self.epochs}] Training")
            # add to log
            if (self.step % self.config['save_summary_steps']) == 0:
                # compute all metrics on this batch
                summary_batch = {metric:self.traffic_metrics[metric](y, y_pred).cpu().detach().item() for metric in self.traffic_metrics}
                summary_batch['loss'] = loss.cpu().detach().item()
                summ.append(summary_batch)
                self.log.add_scalar(f'train/loss', loss.cpu().detach().item(), self.step)
                self.log.add_scalar(f'train/loss_mse', loss_mse.cpu().detach().item(), self.step)
                # for name, param in self.model.named_parameters():
                    # self.log.add_histogram(f'train/{name}', param, self.step)

            loop_postfix['loss'] = loss.cpu().detach().item()
            loop.set_postfix(ordered_dict={k: f'{v:.3f}' for k, v in loop_postfix.items()})

            # tensorboard
            self.step += 1
            gc.collect()
            torch.cuda.empty_cache()

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        print(f"- Train metrics: {metrics_string}")
        return metrics_mean
    def go(self):
        # model_dict = torch.load('./checkpoints/SparseUNet.pk')
        # self.model.load_state_dict(model_dict)

        print(f"Current Task is - {self.model_name}")
        self.version = get_next_version(self.log_dir)
        self.log = SummaryWriter(log_dir=f'{self.log_dir}/version_{self.version}')
        print(f'{self.log_dir}/version_{self.version}')
        self.records = {}
        self.records['metrics'] = []
        self.epochs = self.config['num_epochs']
        best_train_loss = 1e10
        # with tqdm.tqdm(total=self.epochs, ascii=True) as t:
        for epoch in range(1, self.epochs+1):
            self.current_epoch = epoch
            print(f"Epoch {self.current_epoch}/{self.epochs}")
            print('The current LR is', self.optimizer.param_groups[0]["lr"])

            # compute number of batches in one epoch (one full pass over the training set)
            train_metrics = self.train()
            # train_metrics = self.evaluate()
            self.records['metrics'].append([train_metrics])

            # the average log
            for key in train_metrics:
                self.log.add_scalar(f'Avg/{key}', train_metrics[key], epoch)
            
            # scheduler
            # self.scheduler.step()
            self.scheduler.step(train_metrics['loss'])

            # check test loss, save the best model
            self.log.add_scalar(f'Avg/loss', train_metrics['loss'], epoch)
            if train_metrics['loss']<=best_train_loss:
                best_train_loss = train_metrics['loss']
                self.records['best_train_loss'] = best_train_loss
                self.records['best_ep'] = self.current_epoch
                best_ep = self.records['best_ep']
                # save the model state
                torch.save(self.model.state_dict(), f'{self.log_dir}/version_{self.version}/best_model_{best_ep}.pk')
                torch.save(self.scheduler.state_dict(), f'{self.log_dir}/version_{self.version}/best_scheduler_{best_ep}.pk')

            torch.save(self.model.state_dict(), f'{self.log_dir}/version_{self.version}/last_model_{self.current_epoch}.pk')
            torch.save(self.scheduler.state_dict(), f'{self.log_dir}/version_{self.version}/last_scheduler_{self.current_epoch}.pk')

            self.records['epoch'] = self.current_epoch
            self.records['step'] = self.step
            with open(f'{self.log_dir}/version_{self.version}/records.pk', "wb") as f:
                pickle.dump(self.records, f)

            # clean memory
            gc.collect()
            torch.cuda.empty_cache()            
    def goon(self, version=0, epochs=None, mode='last'):

        self.version = version
        # recovery
        self.log = SummaryWriter(log_dir=f'{self.log_dir}/version_{self.version}')
        with open(f'{self.log_dir}/version_{self.version}/records.pk', 'rb') as f:
            self.records = pickle.load(f)
            self.previous_epochs = self.records['epoch']
            self.step = self.records['step'] + 1
            best_train_loss = self.records['best_train_loss']
            best_ep = self.records['best_ep']
            
        # self.scheduler.load_state_dict(torch.load(f'{self.log_dir}/version_{self.version}/scheduler_{best_ep}.pk'))
        if mode=='last':
            self.model.load_state_dict(torch.load(f'{self.log_dir}/version_{self.version}/last_model_{self.previous_epochs}.pk'))
            self.scheduler.load_state_dict(torch.load(f'{self.log_dir}/version_{self.version}/last_scheduler_{self.previous_epochs}.pk'))
        else:
            self.model.load_state_dict(torch.load(f'{self.log_dir}/version_{self.version}/best_model_{best_ep}.pk'))
            self.scheduler.load_state_dict(torch.load(f'{self.log_dir}/version_{self.version}/best_scheduler_{best_ep}.pk'))
            
        if epochs is not None:
            self.epochs = epochs
        else:
            self.epochs = self.config['num_epochs']

        for epoch in range(self.previous_epochs+1, self.previous_epochs+self.epochs+1):
            self.current_epoch = epoch
            # Run one epoch
            print(f"Epoch {self.current_epoch}/{self.epochs + self.previous_epochs}")
            print('The current LR is', self.optimizer.param_groups[0]["lr"])

            # compute number of batches in one epoch (one full pass over the training set)
            train_metrics = self.train()
            self.records['metrics'].append([train_metrics])

            # the average log
            for key in train_metrics:
                self.log.add_scalar(f'Avg/{key}', train_metrics[key], epoch)

            # check test loss, save the best model
            self.log.add_scalars(f'Avg/loss', {'train': train_metrics['loss']}, epoch)
            
            # scheduler
            self.scheduler.step(train_metrics['loss'])
            # self.scheduler.step()
            if train_metrics['loss']<=best_train_loss:
                best_train_loss = train_metrics['loss']
                self.records['best_train_loss'] = best_train_loss
                self.records['best_ep'] = self.current_epoch
                best_ep = self.records['best_ep']
                # save the model state
                torch.save(self.model.state_dict(), f'{self.log_dir}/version_{self.version}/best_model_{best_ep}.pk')
                torch.save(self.scheduler.state_dict(), f'{self.log_dir}/version_{self.version}/best_scheduler_{best_ep}.pk')
                
            torch.save(self.model.state_dict(), f'{self.log_dir}/version_{self.version}/last_model_{self.current_epoch}.pk')
            torch.save(self.scheduler.state_dict(), f'{self.log_dir}/version_{self.version}/last_scheduler_{self.current_epoch}.pk')

            self.records['epoch'] = self.current_epoch
            self.records['step'] = self.step
            with open(f'{self.log_dir}/version_{self.version}/records.pk', "wb") as f:
                pickle.dump(self.records, f)

            # clean memory
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    runner = Runner(model_name='SparseUNet', config=config)
    runner.go()
    # runner.goon()