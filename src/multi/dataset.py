import pandas as pd
import os
from src.utils import load_config, word_map
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser

class MultiIntentDataset(Dataset):
    def __init__(self, data_dir, set_name, mode, cache_dir, max_len=60):
        self.max_len = max_len
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.set_name = set_name
        self.mode = mode
        self.data = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.save_dir = os.path.join(self.cache_dir, self.set_name)
        self.cache_path = os.path.join(self.save_dir, f"multi_{self.mode}.pkl")

        if self.has_cache():
            self.data = self.load()
        else:
            self.data = self.process_data(self.data_dir, set_name, mode, self.tokenizer)
            self.save()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.data.input_ids[index])
        attention_mask = torch.tensor(self.data.attention_mask[index])
        target = torch.tensor(self.data.target[index])
        slot = torch.tensor(self.data.slot[index])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "slot": slot
        }
    
    def has_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        return True
    
    def save(self):
        print("Saving data ...")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data.to_pickle(self.cache_path)
        print("Done.")
        
    def load(self):
        print("Found cached data. Loading ...")
        data = pd.read_pickle(self.cache_path)
        print("Done.")
        return data 
    
    def process_data(self, data_dir, set_name, mode, tokenizer):
        print("No cached data found. Processing data ...")
        data_path = os.path.join(data_dir, set_name, f"{mode}.csv")
        intent_path = os.path.join(data_dir, set_name, "single_intent_dict.csv")
        slot_path = os.path.join(data_dir, set_name, "slot_dict.csv")
        df = pd.read_csv(data_path)

        intent_list = pd.read_csv(intent_path).intent.to_list()
        intent_list.sort()
        intent_dict = dict((intent_list[i], i) for i in range(len(intent_list)))
        
        intent = []
        for i in df.intent:
            multi_intent = i.split("#")
            temp = []
            for j in intent_list:
                if j in multi_intent:
                    temp.append(1)
                else:
                    temp.append(0)
            intent.append(temp)

        slot_list = pd.read_csv(slot_path).slot.to_list()
        slot_list.sort(reverse=True)
        slot_dict = dict((slot_list[i], i) for i in range(len(slot_list)))
        
        input_ids = []
        attention_mask = []
        slot = []
        for index, row in df.iterrows():
            text = row["text"]
            sl = row["slot"].split()
            inputs = tokenizer(
                text = text,
                add_special_tokens = True,
                return_token_type_ids = False,
                padding = "max_length",
                return_attention_mask = True,
                truncation = True,
                max_length = self.max_len,
                return_offsets_mapping = True
            )
            attention_mask.append(inputs['attention_mask'])
            input_ids.append(inputs['input_ids'])
            map = word_map(text)
            temp1 = [sl[map[i[0]]] for i in inputs["offset_mapping"]]

            for ind, token in enumerate(inputs['input_ids']):
                if (token in [100, 101, 102, 0]):
                    temp1[ind] = 'O'

            temp2 = [slot_dict[i] for i in temp1]
            slot.append(temp2)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': intent,
            'slot': slot
        }

        ret = pd.DataFrame(data_dict)
        print("Done.")
        return ret
    
    def getMaxlen(self):
        return self.max_len
    
    def getIntentSlot(self):
        intent_path = os.path.join(self.data_dir, self.set_name, "single_intent_dict.csv")
        slot_path = os.path.join(self.data_dir, self.set_name, "slot_dict.csv")
        intent_list = pd.read_csv(intent_path).intent.to_list()
        slot_list = pd.read_csv(slot_path).slot.to_list()
        return len(intent_list), len(slot_list)
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/local.yaml")
    parser.add_argument("--max_len", type=int, default=60, help="Tokenizer's max length")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    data_dir = config["data_dir"]
    cache_dir = config["cache_dir"]
    dataset_names = config["dataset_names"]
    modes = config["modes"]
    max_len = args.max_len

    print("Using config: ", args.config_path)
    for set_name in dataset_names:
        for mode in modes:
            print(f"Loading dataset: {set_name}-{mode} ...")
            MultiIntentDataset(data_dir, set_name, mode, cache_dir, max_len)