import torch.distributed as dist
import yaml
import json

# Load settings from config.yaml or config.json
config_file = 'config.yaml'  # Change this to 'config.json' if using JSON
with open(config_file, 'r') as file:
    if config_file.endswith('.yaml'):
        config = yaml.safe_load(file)
    elif config_file.endswith('.json'):
        config = json.load(file)
    else:
        raise ValueError("Unsupported configuration file format. Use YAML or JSON.")

# Set the backend (e.g., 'nccl' for NVIDIA GPUs, 'gloo' for CPU)
dist_backend = config['dist_backend']

# Set the world size (number of processes)
world_size = config['world_size']

# Set the rank (unique identifier for each process)
rank = config['rank']

# Set the master address and port for initializing the process group
master_addr = config['master_addr']
master_port = config['master_port']

# Initialize the distributed environment
dist.init_process_group(
    backend=dist_backend,
    world_size=world_size,
    rank=rank,
    init_method=f"tcp://{master_addr}:{master_port}"
)
