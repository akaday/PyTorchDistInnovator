import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(image_dir, batch_size, shuffle=True, transform=None):
    dataset = CustomDataset(image_dir, transform=transform)

    # Create a DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist.is_initialized() else None
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None and shuffle), sampler=sampler)

    return data_loader
