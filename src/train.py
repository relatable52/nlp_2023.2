from collections import defaultdict, OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import IntentDataset

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, max_len):
    model = model.train()
    losses = []
    correct_intent = 0
    correct_slot = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        slots = d["slots"].to(device)
        
        output = model(input_ids, attention_mask)
        intent_logits = output['intent']
        slot_logits = output['slot']
        
        intent_loss = loss_fn(intent_logits.view(-1, 26), targets.view(-1))
        slot_loss = loss_fn(slot_logits.view(-1,129), slots.view(-1))
        loss = slot_loss+intent_loss
        
        _, intent = torch.max(intent_logits, dim=1)
        correct_intent += torch.sum(intent == targets)
        _, slot = torch.max(slot_logits.view(-1, 129), dim=1)
        correct_slot += torch.sum(slot == slots.view(-1))
        
        losses.append(loss.item())
        
        # Backward prop
        loss.backward()
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_intent.double()/n_examples, correct_slot.double()/n_examples/max_len, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples, max_len):
    model = model.eval()
    
    losses = []
    correct_intent = 0
    correct_slot = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            slots = d["slots"].to(device)
            
            # Get model outputs
            output = model(input_ids, attention_mask)
            intent_logits = output['intent']
            slot_logits = output['slot']
            
            intent_loss = loss_fn(intent_logits.view(-1, 26), targets.view(-1))
            slot_loss = loss_fn(slot_logits.view(-1,129), slots.view(-1))
            loss = slot_loss+intent_loss
            
            _, intent = torch.max(intent_logits, dim=1)
            correct_intent += torch.sum(intent == targets)
            _, slot = torch.max(slot_logits.view(-1, 129), dim=1)
            correct_slot += torch.sum(slot == slots.view(-1))
            
            losses.append(loss.item())
            
    return correct_intent.double()/n_examples, correct_slot.double()/n_examples/max_len, np.mean(losses)

def train(epochs, model, loss_fn, optimizer, train_dataset, val_dataset, train_param, val_param, device, scheduler, save_dir, set_name):
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
            loss_fn,optimizer,
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
            loss_fn,
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
        checkpoint = os.path.join(save_dir, set_name)
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

        # If we beat prev performance
        if val_intent_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            best_accuracy = val_intent_acc
            
    torch.save(model.state_dict(), os.path.join(save_dir, 'last.pt'))