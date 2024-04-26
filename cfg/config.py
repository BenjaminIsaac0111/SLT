from pathlib import Path
import yaml


def load_config(file=None):
    config_path = Path(file)
    try:
        with config_path.open() as config:
            try:
                return yaml.safe_load(config)
            except yaml.YAMLError as exc:
                print(exc)
    except FileNotFoundError:
        print('Config file {} was not found.'.format(file))
