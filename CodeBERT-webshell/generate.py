import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from gpt2 import respond_to_batch

device = torch.device('cuda:0')
model = AutoModelWithLMHead.from_pretrained('models/gpt2_sqli_rand4000/')
_ = model.to(device)
_ = model.eval()
tokenizer = AutoTokenizer.from_pretrained('gpt2')

bsz = 16
bos_token = '0'
bos_token_id = tokenizer._convert_token_to_id(bos_token)
with open('choice.txt','a') as f1:
    for j in range(0,1000):
        queries = torch.LongTensor([[bos_token_id]]).expand(bsz, 1)
        queries = queries.to(device)
        outputs = respond_to_batch(model, queries, txt_len=60, top_k=20)
    
        for i in range(len(outputs)):
            tmpstr = tokenizer.decode(outputs[i])
            #print(tmpstr)
            count = tmpstr.find('0!',1)
            #print(tmpstr[:count])
            if count != -1:
                f1.write(tmpstr[:count])
                f1.write('\n')
f1.close()
