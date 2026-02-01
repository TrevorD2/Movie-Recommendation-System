import yaml

def load_config(path='src/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)