# PyTorchDistInnovator

Welcome to PyTorchDistInnovator!

This is a project designed to showcase how PyTorch can be used in conjunction with FastAPI to create an efficient and scalable API.

## Getting Started

Follow these instructions to get your development environment set up.

### Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- Uvicorn

### Installation

Clone the repository:
```bash
git clone https://github.com/akaday/PyTorchDistInnovator.git
cd PyTorchDistInnovator

## Configuration File

The configuration file is used to manage hyperparameters and other settings, making it easier to modify configurations without changing the code. You can use either a YAML file (`config.yaml`) or a JSON file (`config.json`) for the configuration.

### Example `config.yaml` file

```yaml
dist_backend: 'nccl'
world_size: 4
rank: 0
master_addr: 'localhost'
master_port: '12345'

hyperparameters:
  learning_rate: 0.001
  batch_size: 32
```

### Example `config.json` file

```json
{
  "dist_backend": "nccl",
  "world_size": 4,
  "rank": 0,
  "master_addr": "localhost",
  "master_port": "12345",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32
  }
}
```

### Using the Configuration File

1. Create a `config.yaml` or `config.json` file in the root directory of the project.
2. Add the necessary settings and hyperparameters to the configuration file.
3. The code will automatically read the configuration from the file and use the specified settings and hyperparameters.

## Modifying the Configuration File

To modify the configuration file to add new hyperparameters, follow these steps:

1. Open the `config.yaml` or `config.json` file.
2. Add new hyperparameters under the `hyperparameters` section. Ensure that the new hyperparameters are properly indented and follow the YAML or JSON syntax.

### Example of adding new hyperparameters to `config.yaml`

```yaml
dist_backend: 'nccl'
world_size: 4
rank: 0
master_addr: 'localhost'
master_port: '12345'

hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  momentum: 0.9
  weight_decay: 0.0001
```

### Example of adding new hyperparameters to `config.json`

```json
{
  "dist_backend": "nccl",
  "world_size": 4,
  "rank": 0,
  "master_addr": "localhost",
  "master_port": "12345",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "momentum": 0.9,
    "weight_decay": 0.0001
  }
}
```

After adding the new hyperparameters, you can access them in the code by updating the relevant sections in the `distributed_training.py` file. The hyperparameters are loaded from the configuration file and stored in the `hyperparameters` dictionary. You can then use the new hyperparameters in your code as needed.

## Using TensorBoard for Real-Time Monitoring and Visualization

To integrate TensorBoard for real-time monitoring and visualization of training metrics, follow these steps:

1. Install TensorBoard by running `pip install tensorboard`.
2. Import TensorBoard in your `distributed_training.py` file:
   ```python
   from torch.utils.tensorboard import SummaryWriter
   ```
3. Create a `SummaryWriter` instance to log metrics:
   ```python
   writer = SummaryWriter()
   ```
4. Log the training metrics such as loss and accuracy to TensorBoard within the training loop:
   ```python
   writer.add_scalar('Loss/train', loss.item(), epoch)
   ```
5. Close the `SummaryWriter` at the end of the training loop:
   ```python
   writer.close()
   ```

## Using Weights & Biases for Real-Time Monitoring and Visualization

To integrate Weights & Biases (wandb) for real-time monitoring and visualization of training metrics, follow these steps:

1. Install Weights & Biases by running `pip install wandb`.
2. Import Weights & Biases in your `distributed_training.py` file:
   ```python
   import wandb
   ```
3. Initialize a Weights & Biases run to start logging metrics:
   ```python
   wandb.init(project="pytorch-dist-innovator")
   ```
4. Log the training metrics such as loss and accuracy to Weights & Biases within the training loop:
   ```python
   wandb.log({"loss": loss.item()})
   ```
