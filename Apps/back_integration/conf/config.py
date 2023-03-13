import yaml

def get_config(config_file):
    f = open(config_file)
    conf = yaml.safe_load(f)
    return conf



