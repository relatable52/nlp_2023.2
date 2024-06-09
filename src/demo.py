import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.base.dataset import IntentDataset
from src.base.model import JointBertModel
from src.utils import load_config
import os
import pandas as pd

def predict_intent_and_slots(text, intent_list, slot_list, model, tokenizer):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=60
    )

    # print(inputs, type(inputs))
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    intent_logits = outputs['intent']
    slot_logits = outputs['slot']
    
    # Predict intent
    intent_id = torch.argmax(intent_logits, dim=1).item()
    intent = intent_list[intent_id]
    
    # Predict slots
    slot_ids = torch.argmax(slot_logits, dim=2).squeeze().tolist()

    info = ""
    last = ""
    slots = []

    tokens = tokenizer.tokenize(text, add_special_tokens=True)

    for token, slot_id in zip(tokens, slot_ids):
        if slot_id != 0:
            prefix = slot_list[slot_id][:2]
            slot = slot_list[slot_id][2:]
            if token.startswith("##"):
                try:
                    slots[-1][1] += token[2:]
                except:
                    slots.append([slot, last+token[2:]])
            elif prefix == "I-":
                try:
                    slots[-1][1] += " " + token
                except:
                    slots.append([slot, last+" "+token])
            elif prefix == "B-":
                slots.append([slot, token])
        last = token

    for x in slots:
        info += f"{x[0]}: {x[1]}\n"

    return intent, info

# Gradio interface
def gradio_interface(text, data_type, intent_list_atis, slot_list_atis, model_atis, intent_list_snips, slot_list_snips, model_snips, tokenizer):
    intent, slots = predict_intent_and_slots(text, intent_list_atis, slot_list_atis, model_atis, tokenizer) if data_type=='ATIS' else predict_intent_and_slots(text, intent_list_snips, slot_list_snips, model_snips, tokenizer)
    
    return intent, slots

def main():
    # Load configuration
    config = load_config("./config/local.yaml")
    data_dir = config["data_dir"]
    cache_dir = config["cache_dir"]


    # Load atis intent and slot dictionaries
    intent_dict_atis = pd.read_csv(os.path.join(data_dir, "atis", "intent_dict.csv"))
    intent_list_atis = intent_dict_atis['intent'].to_list()

    slot_dict_atis = pd.read_csv(os.path.join(data_dir, "atis", "slot_dict.csv"))
    slot_list_atis = slot_dict_atis['slot'].to_list()
    
    # Load snips intent and slot dictionaries
    intent_dict_snips = pd.read_csv(os.path.join(data_dir, "snips", "intent_dict.csv"))
    intent_list_snips = intent_dict_snips['intent'].to_list()

    slot_dict_snips = pd.read_csv(os.path.join(data_dir, "snips", "slot_dict.csv"))
    slot_list_snips = slot_dict_snips['slot'].to_list()

    # Prepare the atis dataset to get the number of intents and slots
    test_dataset_atis = IntentDataset(data_dir, "atis", "test", cache_dir)
    nintents_atis, nslots_atis = test_dataset_atis.getIntentSlot()

    # Load the atis pre-trained model
    model_dir_atis = os.path.join(config["results_dir"], "atis", "models")
    model_path_atis = os.path.join(model_dir_atis, "best.pt")
    
    model_atis = JointBertModel(nintents_atis, nslots_atis)
    model_atis = nn.DataParallel(model_atis)
    model_atis.load_state_dict(torch.load(model_path_atis, map_location=torch.device('cpu')))
    model_atis.eval()

    # Prepare the snips dataset to get the number of intents and slots
    test_dataset_snips = IntentDataset(data_dir, "snips", "test", cache_dir)
    nintents_snips, nslots_snips = test_dataset_snips.getIntentSlot()

    # Load the snips pre-trained model
    model_dir_snips = os.path.join(config["results_dir"], "snips", "models")
    model_path_snips = os.path.join(model_dir_snips, "best.pt")
    
    model_snips = JointBertModel(nintents_snips, nslots_snips)
    model_snips = nn.DataParallel(model_snips)
    model_snips.load_state_dict(torch.load(model_path_snips, map_location=torch.device('cpu')))
    model_snips.eval()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    iface = gr.Interface(
        fn=lambda text, scope: gradio_interface(text, scope, intent_list_atis, slot_list_atis, model_atis, intent_list_snips, slot_list_snips, model_snips, tokenizer), 
        inputs=[gr.Textbox(label="Input Text"), gr.Dropdown(["ATIS", "SNIPS"], label="Scope")], 
        outputs=[gr.Textbox(label="Predicted Intent"), gr.Textbox(label="Extracted Information")],
        title="Intent Detection and Slot Filling",
        description="Enter a sentence to detect its intent and extract relevant slots (entities)."
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()