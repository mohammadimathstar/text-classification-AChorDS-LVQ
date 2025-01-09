import yaml


def get_config(config_path: str):
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        params = yaml.safe_load(conf_file)

    return params