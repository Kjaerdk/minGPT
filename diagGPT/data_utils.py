from random import shuffle
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, Sampler
from mingpt.utils import CfgNode as CN
from diagGPT.ICD_encoder import get_encoder


class DiagDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.which_mapping = 'chapter'

        return C

    def __init__(self, config, data):
        self.config = config

        e = get_encoder(config.which_mapping)
        self.encoder = e.encoder
        self.decoder = e.decoder

        self.n_diags = len(self.encoder)
        print(f"{self.n_diags} different diagnoses")

        self.data = data  # list of lists (one list being one patient)
        self.config.block_size = max([len(data[p]) for p in range(len(data))]) # block size only used for biases in attention layer

    def get_n_diagnoses(self):
        return self.n_diags

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # diagnoses for the single patient at idx (a list)
        p = self.data[idx]
        # encode every character to an integer
        dix = [self.encoder[d] for d in p]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


class BucketBatchSampler(Sampler):
    """
    Sampler which ensures:
        - Sequences of equal length end up in the same batches
        - Shuffle batches such random which sequence length model is trained on next iteration
    """
    def __init__(self, inputs, batch_size):
        """ inputs: list of lists """
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, len(p)))  # find sequence length of input i
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they are not ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
