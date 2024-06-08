import yaml
from collections import OrderedDict

def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def word_map(text):
    word_count = 0
    map = {}
    for i in range(len(text)):
        map[i] = word_count
        if text[i] == " ":
            word_count += 1
    return map

def mod_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
        
            
            
