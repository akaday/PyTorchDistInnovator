import torch.distributed as dist

# Set the backend (e.g., 'nccl' for NVIDIA GPUs, 'gloo' for CPU)
dist_backend = 'nccl'  # or 'gloo'

# Set the world size (number of processes)
world_size = 4

# Set the rank (unique identifier for each process)
rank = 0

# Set the master address and port for initializing the process group
master_addr = 'localhost'
master_port = '12345'

# Initialize the distributed environment
dist.init_process_group(
    backend=dist_backend,
    world_size=world_size,
    rank=rank,
    init_method=f"tcp://{master_addr}:{master_port}"
)
