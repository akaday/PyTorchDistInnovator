# PyTorchDistInnovator

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
