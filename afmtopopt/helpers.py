from ruamel.yaml import YAML
import os


def load_parameters(filename):
    
    with open(filename, 'r') as f:
        yaml = YAML(typ='safe')
        params = yaml.load(f)
        params['tag'] = os.path.splitext(os.path.basename(f.name))[0]
        params['dir'] = os.path.dirname(f.name)
        return params
