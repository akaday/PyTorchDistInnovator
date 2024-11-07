import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import logging
from distributed_config import dist_backend, world_size, rank, master_addr, master_port
from data_loader import get_data_loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the distributed environment
dist.init_process_group(
    backend=dist_backend,
    world_size=world_size,
    rank=rank,
    init_method=f"tcp://{master_addr}:{master_port}"
)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the CNN model, loss function, and optimizer
model = CNNModel()
model = nn.parallel.DistributedDataParallel(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
data_loader = get_data_loader(image_dir='path/to/images', batch_size=32)

# Example training loop
for epoch in range(10):
    for data in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
    if rank == 0:
        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
