import warnings, os
import torch 
from classifier import CustomRobertaPFClassifier
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
def infer_robertasentencevalidator(contextstr, mdlpath='/content/finetunedrbrta'):
    try:
        ft_model = CustomRobertaPFClassifier()
        mdlpath = mdlpath+'/'
        binpath = mdlpath + 'pytorch_model.bin'
        if os.path.exists(binpath):
            ft_model.load_state_dict(torch.load(binpath))
            ft_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(mdlpath)
            result = {}

            result['contextstr'] = contextstr
            with torch.no_grad():
                #Tokenize inputs
                inputs = tokenizer(contextstr, return_tensors="pt")
                output = ft_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                # get the outputtensor
                evallabel = output[0].argmax().item()
                result['evallabel'] = evallabel
                if evallabel==1:
                  result['prediction'] = 'offensive'
                else:
                  result['prediction'] = 'non-offensive'
            return result

    except Exception as e:
        return (f"Encountered error while performing inference: {e}")