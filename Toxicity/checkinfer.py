import numpy as np 
import random
import torch, os
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import PyTorchModelHubMixin

class CustomRobertaTXClassifier(nn.Module,PyTorchModelHubMixin):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.config = self.roberta.config
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 2)
        self.gelu = nn.GELU()
        self.config.to_json_file('config.json')

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        '''
        Applies a forward function by passsing the input to roberta, extracts relevant embeedings and passes them to classifierlayers
        for calculation of logits
        '''
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

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def rbrtatoxiccnfthrvalidator(contextstr, mdlpath= "/content/drive/MyDrive/rbrtftmdlnew", threshold=0.020, seed_value=42):
    try:
        set_seed(seed_value)
        ft_model = CustomRobertaTXClassifier()
        binpath = mdlpath + '/pytorch_model.bin'
        threshold = str(threshold)
        if os.path.exists(binpath):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ft_model.load_state_dict(torch.load(binpath, map_location=device))
            ft_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(mdlpath +'/')
            result = {}
            result['contextstr'] = contextstr
            with torch.no_grad():
                #Tokenize inputs
                inputs = tokenizer(contextstr, return_tensors="pt")
                output = ft_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                
                cnf_intr = torch.abs(abs(output[0][0][1]) - abs(output[0][0][0]))
                cnf_intrstr = f"{cnf_intr.item():.3f}"
                #print(f"Abs diff confidence interval: {cnf_intr.item():.3f}")
                # get the outputtensor
                evallabel = output[0].argmax().item() 
                cnfthreshold_analysis = (cnf_intrstr>=threshold)
                result['cnf_intr'] = cnf_intrstr
                result['evallabel'] = evallabel
                result['cntthreholdanalysis'] = cnfthreshold_analysis

                if cnfthreshold_analysis:  
                  if evallabel==1:
                    result['prediction'] = 'Toxic'
                  else:
                    result['prediction'] = 'Non-toxic'
                  return result 
                else: 
                  result['prediction'] = 'Uncertain'
                  return result
    except Exception as e:
        return (f"Encountered error while performing inference: {e}")