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

# Create your model, loss function, and optimizer
model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
data_loader = get_data_loader(batch_size=32)

# Example training loop
for epoch in range(10):
    for data, labels in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    if rank == 0:
        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
