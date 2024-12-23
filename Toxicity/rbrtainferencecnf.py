import warnings, os
import torch 
from classifier import CustomRobertaPFClassifier
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

def infer_toxiccnfvalidator(contextstr, mdlpath=OUTPUT_DIR):
    try:
        ft_model = CustomRobertaPFClassifier()
        binpath = mdlpath + '/pytorch_model.bin'
        if os.path.exists(binpath):
            ft_model.load_state_dict(torch.load(binpath))
            ft_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR +'/')
            result = {}
            result['contextstr'] = contextstr
            with torch.no_grad():
                #Tokenize inputs
                inputs = tokenizer(contextstr, return_tensors="pt")
                output = ft_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

                cnf_intr = torch.abs(output[0][1] - output[0][0])
                print(f"Absolute difference: {cnf_intr.item():.4f}")

                # get the outputtensor
                evallabel = output[0].argmax().item()

                result['evallabel'] = evallabel
                if evallabel==1:
                  result['prediction'] = 'toxic'
                else:
                  result['prediction'] = 'non-toxic'
            return result

    except Exception as e:
        return (f"Encountered error while performing inference: {e}")
