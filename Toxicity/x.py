
import torch.nn.functional as f 
from transformers import RobertaModel 
class RobertaToxicityClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ToxicityClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear1 = f.Linear(768, 512)
        self.linear2 = f.Linear(512, 256)
        self.linear3 = f.Linear(256, 128) 
        self.linear4 = f.Linear(128, 2)
        self.gelu = f.GELU() 

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        # Modified to use gelu activation and store output correctly
        x = torch.f.functional.gelu(self.linear1(x)) 
        x = torch.f.functional.gelu(self.linear2(x))
        x = torch.f.functional.gelu(self.linear3(x))
        logits = self.linear4(x) 

        loss = None
        if labels is not None:  
            if class_weights is not None:
                loss_fct = f.CrossEntropyLoss(weight=class_weights)  
            else:
                loss_fct = f.CrossEntropyLoss()  
            loss = loss_fct(logits, labels)

        return logits, loss 
