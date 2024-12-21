from copy import deepcopy
import torch 
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience # the number of epochs to wait for improvement before stopping, default=3
        self.min_delta = min_delta # the minimum change in validation loss that is considered an improvement, default is 0
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            # perfom deep copy instead of shallow which creates tensors that preserves the independent state of model instead of creating a dict maintaining references to tensor objects
            self.best_state_dict = deepcopy(model.state_dict())
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = deepcopy(model.state_dict())


class CustomRobertaPFClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        #self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        #self.dropout2 = nn.Dropout(dropout_rate)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 2)
        self.gelu = nn.GELU()


    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        '''
        Applies a forward function by passsing the input to roberta, extracts relevant embeedings and passes them to classifierlayers
        for calculation of logits
        '''
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        #x = self.dropout1(x)
        x = torch.nn.functional.gelu(self.linear1(x))
        x = torch.nn.functional.gelu(self.linear2(x))
        x = torch.nn.functional.gelu(self.linear3(x))
        #x = self.dropout2(x)
        #return self.linear2(x)
        logits = self.linear4(x)
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss
