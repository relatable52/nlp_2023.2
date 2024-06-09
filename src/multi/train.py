from collections import defaultdict, OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from src.multi.dataset import MultiIntentDataset
from src.utils import load_config
from argparse import ArgumentParser
from src.base.model import JointBertModel
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_epoch(model, data_loader, intent_loss_fn, slot_loss_fn, optimizer, device, scheduler, n_examples, max_len):
    model = model.train()
    losses = []
    correct_intent = 0
    correct_slot = 0
    
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)
        slots = d["slot"].to(device)
        
        output = model(input_ids, attention_mask)
        intent_logits = output['intent']
        slot_logits = output['slot']
        
        intent_loss = intent_loss_fn(intent_logits, targets.float())
        slot_loss = slot_loss_fn(slot_logits.flatten(0, 1), slots.flatten(0, 1))
        loss = slot_loss+intent_loss
        
        intent = torch.where(F.sigmoid(intent_logits)>0.5, 1.0, 0.0)
        correct_intent += torch.sum(intent.flatten() == targets.flatten())/intent_logits.shape[1]
        _, slot = torch.max(slot_logits.flatten(0, 1), dim=1)
        correct_slot += torch.sum(slot == slots.flatten(0, 1))
        
        losses.append(loss.item())
        
        # Backward prop
        loss.backward()
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_intent.double()/n_examples, correct_slot.double()/n_examples/max_len, np.mean(losses)

def eval_model(model, data_loader, intent_loss_fn, slot_loss_fn, device, n_examples, max_len):
    print("Evaluating model...")
    model = model.eval()
    
    losses = []
    correct_intent = 0
    correct_slot = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            slots = d["slot"].to(device)
            
            # Get model outputs
            output = model(input_ids, attention_mask)
            intent_logits = output['intent']
            slot_logits = output['slot']
            
            intent_loss = intent_loss_fn(intent_logits, targets.float())
            slot_loss = slot_loss_fn(slot_logits.flatten(0,1), slots.flatten(0,1))
            loss = slot_loss+intent_loss
            
            intent = torch.where(F.sigmoid(intent_logits)>0.5, 1.0, 0.0)
            correct_intent += torch.sum(intent.flatten() == targets.flatten())/intent_logits.shape[1]
            _, slot = torch.max(slot_logits.flatten(0, 1), dim=1)
            correct_slot += torch.sum(slot == slots.flatten(0, 1))
            
            losses.append(loss.item())
            
    return correct_intent.double()/n_examples, correct_slot.double()/n_examples/max_len, np.mean(losses)

def train(epochs, model, intent_loss_fn, slot_loss_fn, optimizer, train_dataset, val_dataset, train_param, val_param, device, scheduler, save_dir, set_name):
    train_loader = DataLoader(train_dataset, **train_param)
    val_loader = DataLoader(val_dataset, **val_param)
    
    model.to(device)    
    history = defaultdict(list)
    best_accuracy = 0

    torch.cuda.empty_cache() 
    for epoch in range(epochs):
        
        # Show details 
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        intent_acc, slot_acc, train_loss = train_epoch(
            model,
            train_loader,
            intent_loss_fn,
            slot_loss_fn,
            optimizer,
            device,
            scheduler, 
            n_examples=len(train_dataset),
            max_len=train_dataset.getMaxlen()
        )
        
        print(f"Train loss {train_loss}, intent accuracy {intent_acc}, slot accuracy {slot_acc}")
        
        # Get model performance (accuracy and loss)
        val_intent_acc, val_slot_acc, val_loss = eval_model(
            model,
            val_loader,
            intent_loss_fn,
            slot_loss_fn,
            device,
            n_examples=len(val_dataset),
            max_len=val_dataset.getMaxlen()
        )
        
        print(f"Val loss {val_loss}, intent accuracy {val_intent_acc}, slot accuracy {val_slot_acc}")
        print()
        
        history['train_acc'].append(intent_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_intent_acc)
        history['val_loss'].append(val_loss)
        
        # Checkpoint for saving models
        save_path = os.path.join(save_dir, set_name)
        checkpoint = os.path.join(save_path, "models")
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

        # If we beat prev performance
        if val_intent_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(checkpoint, 'multi_best.pt'))
            best_accuracy = val_intent_acc
            
    torch.save(model.state_dict(), os.path.join(checkpoint, 'multi_last.pt'))
    
    train_acc = [float(i.cpu()) for i in history['train_acc']]
    val_acc = [float(i.cpu()) for i in history['val_acc']]
    train_loss = [float(i) for i in history['train_loss']]
    val_loss = [float(i) for i in history['val_loss']]
    his_df = pd.DataFrame({
        "train_acc": train_acc,
        "val_acc": val_acc, 
        "train_loss": train_loss, 
        "val_loss": val_loss
    })
    his_df.to_csv(os.path.join(save_path, "multi_training_log.csv"))

    # Plot training and validation accuracy
    plt.plot(his_df.train_acc, label='train accuracy')
    plt.plot(his_df.val_acc, label='validation accuracy')

    # Graph chars
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);
    plt.savefig(os.path.join(save_path, "multi_accuracy.pdf"))
    plt.close()
    plt.clf()

    # Plot training and validation loss
    plt.plot(his_df.train_loss, label='train loss')
    plt.plot(his_df.val_loss, label='validation loss')

    # Graph chars
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_path, "multi_loss.pdf"))
    plt.close()
    plt.clf()

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/local.yaml")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="atis")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    config = load_config(args.config_path)
    data_dir = config["data_dir"]
    cache_dir = config["cache_dir"]
    results_dir = config["results_dir"]
    epochs = args.epochs
    dataset = args.dataset

    TRAIN_PARAM = {"batch_size":10, "shuffle":True, "num_workers":2}
    TEST_PARAM = {"batch_size":10, "shuffle":True, "num_workers":2}

    train_dataset = MultiIntentDataset(data_dir, dataset, "train", cache_dir)
    test_dataset = MultiIntentDataset(data_dir, dataset, "test", cache_dir)

    nintents, nslots = train_dataset.getIntentSlot()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointBertModel(nintents, nslots)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    intent_loss_fn = torch.nn.BCEWithLogitsLoss()
    slot_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    total_steps = (len(train_dataset)//TRAIN_PARAM["batch_size"]+1) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train(epochs, model, intent_loss_fn, slot_loss_fn, optimizer, train_dataset, test_dataset, TRAIN_PARAM, TEST_PARAM, device, scheduler, results_dir, dataset)

if __name__ == "__main__":
    main()