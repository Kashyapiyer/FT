import warnings, os
import torch 
from classifier import CustomRobertaPFClassifier
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

def rbrtatoxiccnfthrvalidator(contextstr, mdlpath=OUTPUT_DIR, threshold=0.020):
    try:
        ft_model = CustomRobertaPFClassifier()
        binpath = mdlpath + '/pytorch_model.bin'
        threshold = str(threshold)
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

                print(output[0][0][0], output[0][0][1])
                cnf_intr = torch.abs(abs(output[0][0][1]) - abs(output[0][0][0]))

                cnf_intrstr = f"{cnf_intr.item():.3f}"

                print(f"Abs diff confidence interval: {cnf_intr.item():.3f}")

                # get the outputtensor
                evallabel = output[0].argmax().item()

              
                cnfthreshold_analysis = (cnf_intrstr>=threshold)
                result['cnf_intr'] = cnf_intrstr
                result['evallabel'] = evallabel
                result['cntthreholdanalysis'] = cnfthreshold_analysis

                if cnfthreshold_analysis: 
                   
                  if evallabel==1:
                    result['prediction'] = 'toxic'
                  else:
                    result['prediction'] = 'non-toxic'
                  return result 
                else: 
                  result['prediction'] = 'uncertain'
                  return result
    except Exception as e:
        return (f"Encountered error while performing inference: {e}")
