import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
import scipy.io
import numpy as np
import random
from params import std, mean
import time

DEFAULT_ROOT = '/scratch/sagar/slf/train_set/set_harsh_torch_raw_unnormalized/slf_mat'

class SLF(Dataset):
    def __init__(self, root=DEFAULT_ROOT, train=True, download=True, transform=None, total_data=None, sampling=False, normalize=True):
        self.root_dir = root
        if not total_data is None:
            self.num_examples = total_data
        else:
            if train == True:
                self.num_examples = 500000
            else:
                self.num_examples = 2000
        self.sampling = sampling
        sample_size = [0.01,0.30]
        self.sampling_rate = sample_size[1] - sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        self.normalize = normalize
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir,
                                str(idx)+'.pt')
        sample = torch.load(filename)
        if self.sampling:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,51,51), dtype=torch.float32)
            subsample = sample*bool_mask
            return subsample, sample
        
        if self.normalize:
            sample = np.log(sample)
            sample = sample/sample.min()
        return sample


class SLFDataset(Dataset):
    """SLF loader"""

    def __init__(self, root_dir, csv_file=None, transform=None, total_data=None, normalize=True, sample_size=[0.01,0.20], fixed_size=None, fixed_mask=False, no_sampling=False):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            total_data: Number of data points
            normalize: Whether to normalize such that the largest value is 1
            sample_size: range off sampling percentage
            fixed_size: if not none, fixed_size will be used as the sampling size
            fixed_mask: if true, the same mask will be used 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.NUM_SAMPLES = int(0.20*51*51)
        self.nrow, self.ncol = (51, 51)
        if not total_data is None:
            self.num_examples = total_data
        else:
            self.num_examples = 500000
        self.sampling_rate = sample_size[1]-sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        
        if fixed_size:
            self.sampling_rate = 0
            self.omega_start_point = 1.0 - fixed_size
        
        self.fixed_mask = fixed_mask
        self.no_sampling = no_sampling
        if self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            self.bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            self.int_mask = self.bool_mask*torch.ones((1,51,51), dtype=torch.float32)
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir,
                                str(idx)+'.pt')
        sample = torch.load(filename)

        if self.no_sampling:
            return sample
        
        if not self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,51,51), dtype=torch.float32)
            sampled_slf = sample*bool_mask
        else:
            int_mask = self.int_mask
            sampled_slf = sample*self.bool_mask
        
        return torch.cat((int_mask,sampled_slf), dim=0), sample

class SLFDatasetUnsampled(Dataset):
    """SLF loader"""

    def __init__(self, root_dir, csv_file=None, transform=None, total_data=None, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            total_data: Number of data points
            normalize: Whether to normalize such that the largest value is 1
        """
        self.root_dir = root_dir
        if not total_data is None:
            self.num_examples = total_data
        else:
            self.num_examples = 500000
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir,
                                str(idx)+'.pt')
        sample = torch.load(filename)
        return sample
    

class SLFDatasetMat(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, transform=None, total_data=None, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.params = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.nrow, self.ncol = (51, 51)
        self.NUM_SAMPLES = int(0.20*51*51)
        self.num_examples = len(self.params)
        if not total_data is None:
            self.num_examples = total_data
        self.normalize = normalize

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.root_dir,
                                self.params.iloc[idx, 0])
        
        slf_data = scipy.io.loadmat(filename)

        
        
        slf = slf_data['Sc']
        
        # take log 
        true_slf = torch.tensor(np.log10(slf), dtype=torch.float32)
#         true_slf = torch.tensor(slf, dtype=torch.float32)
        
        
        if self.normalize:
            true_slf = true_slf/true_slf.min()
        
        # sampling 
        sample_idx = random.sample(range(51*51), self.NUM_SAMPLES )

        mask = torch.ones(true_slf.shape)
        r,c = [(i // self.ncol) for i in sample_idx] , [(i % self.ncol) for i in sample_idx]

        sampled_slf = torch.zeros(true_slf.shape, dtype=torch.float32)
        for i in range(len(r)):
            mask[r[i],c[i]] = 0
            sampled_slf[r[i],c[i]] = true_slf[r[i],c[i]]

        # Get data
        true_slf = true_slf.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        sampled_slf = sampled_slf.unsqueeze(dim=0)
        sampled_slf = torch.cat((mask, sampled_slf), dim=0)
        
#         sample = {'sampled_slf': sampled_slf, 'true_slf':true_slf}
        
        
#         if self.transform:
#             sample = self.transform(sample)
        dt = torch.cat((sampled_slf, true_slf), dim=0)

#         dt = data
#         torch.save(data,'data.pt')
#         dt = torch.load('data.pt')
        
#         return dt[0:2], dt[2]

        return sampled_slf, true_slf


class SLFDatasetMatTrue(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, raw_format = False, transform=None, total_data=None, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.params = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.nrow, self.ncol = (51, 51)
        self.NUM_SAMPLES = int(0.20*51*51)
        self.num_examples = len(self.params)
        if not total_data is None:
            self.num_examples = total_data
        self.normalize = normalize
        self.raw_format = raw_format

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.root_dir,
                                self.params.iloc[idx, 0])
        
        slf_data = scipy.io.loadmat(filename)

        slf = slf_data['Sc']
        
        # take log 
        if not self.raw_format:
            true_slf = torch.tensor(np.log10(slf + 1e-16), dtype=torch.float32)
        else:
            true_slf = torch.tensor(slf, dtype=torch.float32)
        
        
        if self.normalize:
            if not self.raw_format:
                true_slf = true_slf/true_slf.min()
            else:
                true_slf = true_slf/(true_slf.max()+1e-16)
        
        return true_slf.unsqueeze(dim=0)


