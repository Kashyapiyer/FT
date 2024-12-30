import torch 
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

class RbrtaToxicityClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 2)
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        x = torch.nn.functional.gelu(self.linear1(x))
        x = torch.nn.functional.gelu(self.linear2(x))
        x = torch.nn.functional.gelu(self.linear3(x))
        logits = self.linear4(x)
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return logits, loss
