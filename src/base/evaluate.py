from argparse import ArgumentParser
from src.utils import load_config, mod_state_dict
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from src.base.dataset import IntentDataset
from src.base.model import JointBertModel
from torch.utils.data import DataLoader

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/local.yaml")
    parser.add_argument("--set_name", type=str, default="atis")
    return parser.parse_args()

def get_predictions(model, data_loader):
    model = model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    intent_predictions = []
    intent_prediction_probs = []
    slot_predictions = []
    slot_prediction_probs = []
    real_intent_values = []
    real_slot_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            intent_targets = d["target"].to(device)
            slot_targets = d["slot"].to(device)

            # Get outouts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Intent detection predictions
            _, intent_pred = torch.max(outputs['intent'], dim=1)

            # Slot filling predictions
            _, slot_pred = torch.max(outputs['slot'], dim=2)

            intent_predictions.extend(intent_pred)
            intent_prediction_probs.extend(outputs['intent'])
            slot_predictions.extend(slot_pred)
            slot_prediction_probs.extend(outputs['slot'])
            real_intent_values.extend(intent_targets)
            real_slot_values.extend(slot_targets)

    intent_predictions = torch.stack(intent_predictions).cpu()
    intent_prediction_probs = torch.stack(intent_prediction_probs).cpu()
    slot_predictions = torch.stack(slot_predictions).cpu()
    slot_prediction_probs = torch.stack(slot_prediction_probs).cpu()
    real_intent_values = torch.stack(real_intent_values).cpu()
    real_slot_values = torch.stack(real_slot_values).cpu()

    return intent_predictions, intent_prediction_probs, slot_predictions, slot_prediction_probs, real_intent_values, real_slot_values

def show_confusion_matrix(intent_confusion_matrix, save_dir, title, name):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    hmap = sns.heatmap(intent_confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, name))
    plt.show()

def evaluate_then_save(model, data_loader, set_name, save_dir):
    save_path = os.path.join(save_dir, set_name)
    y_intent_pred, y_intent_pred_probs, y_slot_pred, y_slot_pred_probs, y_intent_test, y_slot_test = get_predictions(
        model,
        data_loader
    )

    intent_list = pd.read_csv(os.path.joint("./data/atis_snips_v2/", set_name, "intent_dict.csv")).intent.to_list()
    slot_list = pd.read_csv(os.path.joint("./data/atis_snips_v2/", set_name, "slot_dict.csv")).slot.to_list()

    intent_report = classification_report(y_intent_test, y_intent_pred, target_names=intent_list, output_dict=True)
    intent_df = pd.DataFrame(intent_report).transpose()

    intent_test_labels = pd.DataFrame(y_intent_test.numpy())
    intent_test_labels = list(np.sort(intent_test_labels[0].unique()))
    intent_cm = confusion_matrix(y_intent_test, y_intent_pred, labels=intent_test_labels)
    intent_df_cm = pd.DataFrame(intent_cm, index=intent_test_labels, columns=intent_test_labels)

    print("Intent report:")
    print(intent_df)
    intent_df.to_csv(os.path.join(save_path, "intent_report.csv"))
    
    print("Confusion matrix:")
    show_confusion_matrix(intent_df_cm, save_path, "Intents confusion matrix", "intent_cm.pdf")
    intent_df_cm.to_csv(os.path.join(save_path, "intent_cm.csv"))

    y_slot_pred_flat = y_slot_pred.view(-1)
    y_slot_test_flat = y_slot_test.view(-1)
    slot_report = classification_report(y_slot_test_flat, y_slot_pred_flat, target_names=slot_list, output_dict=True)
    slot_df = pd.DataFrame(slot_report).transpose()
    
    slot_test_labels = pd.DataFrame(y_slot_test.numpy())
    slot_test_labels = list(np.sort(slot_test_labels[0].unique()))
    slot_cm = confusion_matrix(y_slot_test_flat, y_slot_pred_flat, labels=slot_test_labels)
    slot_df_cm = pd.DataFrame(slot_cm, index=slot_test_labels, columns=slot_test_labels)

    print("Slot report:")
    print(slot_df)
    slot_df.to_csv(os.path.join(save_path, "slot_report.csv"))

    print("Confusion matrix:")
    show_confusion_matrix(slot_df_cm, save_path, "Slots confusion matrix", "slot_cm.pdf")
    slot_df_cm.to_csv(os.path.join(save_path, "slot_cm.csv"))


def main():
    args = get_args()
    config = load_config(args.config_path)
    data_dir = config["data_dir"]
    cache_dir = config["cache_dir"]
    save_dir = config["results_dir"]

    test_dataset = IntentDataset(data_dir, args.set_name, "test", cache_dir)
    nintents, nslots = test_dataset.getIntentSlot()
    model = JointBertModel(nintents, nslots)

    model_path = os.path.join(save_dir, args.set_name, "models", "last.pt")
    model.load_state_dict(mod_state_dict(torch.load(model_path)))

    PARAM = {"batch_size":10, "shuffle":True, "num_workers":2}
    test_dataloader = DataLoader(test_dataset, **PARAM)

    evaluate_then_save(model, test_dataloader, args.set_name, save_dir)

if __name__ == "__main__":
    main()











