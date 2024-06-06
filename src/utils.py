import yaml

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
        
            
            
