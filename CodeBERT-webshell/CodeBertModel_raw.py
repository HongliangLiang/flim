import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import transformers
import torch.nn.functional as F

class TextCNNClassifer(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer, self).__init__()
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (1, 2, 3, 4, 6, 8)])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * len((1, 2, 3, 4, 6, 8)), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, ids, mask, token_type_ids):
        print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        h = self.encode(ids, attention_mask=mask, token_type_ids=token_type_ids)
        print('h shape :',h.shape)
        # only use the first h in the sequence
        out = h['last_hidden_state']
        print('out shape :', out.shape)
        out = out.unsqueeze(1)
        print('out unsqueeze shape :', out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        print('out shape :', out.shape)
        out = self.dropout(out)
        print('out shape :', out.shape)
        out = self.fc(out)
        print('最终的输出shape :',out.shape)
        return out
if __name__ == '__main__':
    model = TextCNNClassifer()
    ids=[]
    mask=[]
    token_type_ids=[]
    res=model(ids,mask,token_type_ids)
