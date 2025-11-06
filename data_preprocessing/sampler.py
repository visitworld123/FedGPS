import torch
from torch.utils.data import Sampler,DataLoader


class RepeatRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n).tolist()  # 随机打乱样本索引
        while True:
            yield from indices

    def __len__(self):
        return len(self.data_source)