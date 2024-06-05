from transformers import BertModel
from torch import nn

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class JointBertModel(nn.Module):
    def __init__(self, nintents, nslots):
        super(JointBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.intent_classifier = IntentClassifier(input_dim=self.bert.config.hidden_size, num_intent_labels=nintents, dropout_rate=0.05)
        self.slot_classifier = SlotClassifier(input_dim=self.bert.config.hidden_size, num_slot_labels=nslots, dropout_rate=0.05)
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output[0]
        pooled_output = output[1]
        
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return {
            'intent': intent_logits,
            'slot': slot_logits
        }
        