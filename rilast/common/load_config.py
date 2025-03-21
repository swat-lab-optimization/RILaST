import os
import yaml
def load_config(class_name):
    """Load the YAML config file for a specific class."""
    config_path = os.path.join("rilast\\test_generators\\configs", f"{class_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config