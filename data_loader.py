import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data_loader(batch_size, shuffle=True):
    # Generate some random data for demonstration purposes
    data = torch.randn(100, 10)
    labels = torch.randn(100, 1)

    dataset = CustomDataset(data, labels)

    # Create a DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist.is_initialized() else None
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None and shuffle), sampler=sampler)

    return data_loader
