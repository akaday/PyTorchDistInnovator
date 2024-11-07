import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from distributed_config import dist_backend, world_size, rank, master_addr, master_port

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
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(torch.randn(10))
    loss = loss_fn(output, torch.randn(1))
    loss.backward()
    optimizer.step()
    if rank == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
