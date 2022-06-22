import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import transformers
import torch.nn.functional as F
class TextCNNClassifer_coscross(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer_coscross, self).__init__()
        
#         self.encode.config.type_vocab_size=2
        
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(768, 2)#768*2+
#         self.config=config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        #(32, 1, 40, 300)
       
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")
#         print(self.encode.config)

    def forward(self, input_ids,input_mask,feature):
        # print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        out_h = self.encode(input_ids, attention_mask=input_mask)
        # print('h shape :',h[0].shape,h[1].shape)
        # only use the first h in the sequence
        # out = h['last_hidden_state']
        bert_out=out_h[1]
        bert_out=self.dropout(bert_out)
#         print('feature shape :',feature.shape)
#         print('bert_out shape :',bert_out.shape)
#         print('con_out shape:',con_out.shape)
        out = self.fc(bert_out)
        # print('最终的输出shape :',out.shape)
        return out
class TextCNNClassifer_cosloss(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer_cosloss, self).__init__()
        
#         self.encode.config.type_vocab_size=2
        
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(768, 2)#768*2+
#         self.config=config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        #(32, 1, 40, 300)
       
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")
#         print(self.encode.config)
        self.embedding_loss=torch.nn.CosineEmbeddingLoss(margin=0.5, size_average=None, reduce=None, reduction='mean')
    def forward(self, code_input_ids,code_input_mask,nl_input_ids,nl_input_mask,targets,ttype='train'):
        # print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        bs=code_input_ids.shape[0]
        input_ids=torch.cat((code_input_ids,nl_input_ids),0)
        input_masks=torch.cat((code_input_mask,nl_input_mask),0)
        out_h = self.encode(input_ids, attention_mask=input_masks)
        # print('h shape :',h[0].shape,h[1].shape)
        # only use the first h in the sequence
        # out = h['last_hidden_state']
        bert_out=out_h[1]
        code_vec=bert_out[:bs]
        nl_vec=bert_out[bs:]
        
        print('code_vec shape :',code_vec.shape)
        print('nl_vec shape :',nl_vec.shape)
        
#         print('con_out shape:',con_out.shape)
        loss=self.embedding_loss(code_vec,nl_vec,targets)
        # print('最终的输出shape :',out.shape)
        return loss,code_vec,nl_vec
class TextCNNClassifer_pair_nofeature(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer_pair_nofeature, self).__init__()
        
#         self.encode.config.type_vocab_size=2
        
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(768, 2)#768*2+
#         self.config=config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        #(32, 1, 40, 300)
       
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")
#         print(self.encode.config)

    def forward(self, input_ids,input_mask,feature):
        # print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        out_h = self.encode(input_ids, attention_mask=input_mask)
        # print('h shape :',h[0].shape,h[1].shape)
        # only use the first h in the sequence
        # out = h['last_hidden_state']
        bert_out=out_h[1]
        bert_out=self.dropout(bert_out)
#         print('feature shape :',feature.shape)
#         print('bert_out shape :',bert_out.shape)
#         print('con_out shape:',con_out.shape)
        out = self.fc(bert_out)
        # print('最终的输出shape :',out.shape)
        return out
class TextCNNClassifer_pair(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer_pair, self).__init__()
        
#         self.encode.config.type_vocab_size=2
        
        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(768+19, 2)#768*2+
#         self.config=config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        #(32, 1, 40, 300)
       
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")
#         print(self.encode.config)

    def forward(self, input_ids,input_mask,feature):
        # print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        out_h = self.encode(input_ids, attention_mask=input_mask)
        # print('h shape :',h[0].shape,h[1].shape)
        # only use the first h in the sequence
        # out = h['last_hidden_state']
        bert_out=out_h[1]
        
#         print('feature shape :',feature.shape)
#         print('bert_out shape :',bert_out.shape)
        con_out=torch.cat((bert_out,feature),axis=1)
#         print('con_out shape:',con_out.shape)
        out = self.fc(con_out)
        # print('最终的输出shape :',out.shape)
        return out
class TextCNNClassifer(torch.nn.Module):
    def __init__(self,config_hdj):
        super(TextCNNClassifer, self).__init__()
        
#         self.encode.config.type_vocab_size=2
        
        self.config_hdj=config_hdj
        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(300*2, 2)#768*2+
#         self.config=config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        #(32, 1, 40, 300)
        self.convs = nn.ModuleList([
            nn.Sequential(
                          nn.Conv2d(in_channels=config_hdj.in_channels, out_channels=config_hdj.out_channels, kernel_size=(h,config_hdj.embedding_size ), stride=(1, 1), bias=True),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(config_hdj.wordNums-h+1, 1), stride=(1, 1))
            )
#                           nn.MaxPool1d(kernel_size=config.max_text_len - h + 1))
            for h in config_hdj.window_sizes
        ])
        #(32, 1, 100,300)
        self.convs_lines = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=config_hdj.in_channels, out_channels=config_hdj.out_channels, kernel_size=(h, config_hdj.out_channels*len(config_hdj.window_sizes)), stride=(1, 1), bias=True),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(config_hdj.codeLineNums-h+1, 1), stride=(1, 1))
                         )
            for h in config_hdj.window_sizes
        ])
    
        self.report_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=config_hdj.in_channels, out_channels=config_hdj.out_channels, kernel_size=(h, config_hdj.embedding_size), stride=(1, 1), bias=True),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(config_hdj.reportLineNums-h+1, 1), stride=(1, 1))
                         )
            for h in config_hdj.window_sizes
        ])
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.fc768_300 = nn.Linear(in_features=768,out_features=300,bias=True)
        print(self.encode.config)

    def forward(self, report_ids,report_mask, code_ids,code_mask,cnn_report,cnn_code):
        # print('三个输入的shape :',ids.shape,mask.shape,token_type_ids.shape)
        report_h = self.encode(report_ids, attention_mask=report_mask)
        code_h=self.encode(code_ids, attention_mask=code_mask)
        # print('h shape :',h[0].shape,h[1].shape)
        # only use the first h in the sequence
        # out = h['last_hidden_state']
        bert_report_out=report_h[1]
        bert_code_out=code_h[1]
        
        embed_x = cnn_code.view(-1,1,self.config_hdj.wordNums,self.config_hdj.embedding_size)#[32*100,40,300]-->[32*100,1,40,300]
#         embed_y=y_report.view(-1,1,self.config.reportLineNums,config.out_channels*len(config.window_sizes))#[32,150,300]-->[32,1,150,300]
#         embed_x = x_code.view(-1,1,self.config.wordNums,config.embedding_size)#[32*100,40,300]-->[32*100,1,40,300]
        embed_y=cnn_report.view(-1,1,self.config_hdj.reportLineNums,self.config_hdj.embedding_size)#[32,150,300]-->[32,1,150,300]

        out = [conv(embed_x) for conv in self.convs]  #[32*100,1,40,300]--> [32*100,100,38,1]--->pool [32*100,100,1,1]
        out = torch.cat(out, dim=1)
#         print('code第一次拼接的shape',out.shape)#[32*100, 300, 1, 1]
        out = out.view(-1, 1, self.config_hdj.codeLineNums,self.config_hdj.out_channels*len(self.config_hdj.window_sizes))#[32,1,100,300]
#         out = out.view(-1, 1, config.codeLineNums,config.embedding_size)#[32,1,100,300]
#         print('code第一次卷积',out.shape)

        # 开始在行间做二次卷积
        out2 = [conv(out) for conv in self.convs_lines]#[32,1,100,300]-->[32,100,98,1]-->pool[32,100,1,1]
#         for o in out2:
#             print('o2', o.size())  # ([32, 100, 1, 1])
        codeOutMerge = torch.cat(out2, dim=1)  # [32,300,1,1]
#         print('code二次卷积size：',outMerge.size())  # [32,300,1,1]
        codeOutMerge = codeOutMerge.view(-1, codeOutMerge.size(1))
    
        report_out = [conv(embed_y) for conv in self.report_convs]  # [32,1,150,300]-->[32,100,148,1]-->pool[32,100,1,1]
        report_outMerge = torch.cat(report_out, dim=1)#[32,300,1,1]
#         print('report 卷积size:',report_outMerge.size())
        report_outMerge = report_outMerge.view(-1, report_outMerge.size(1))#
    
        bert_report_out=self.fc768_300(bert_report_out)
        report_add=torch.add(bert_report_out,report_outMerge)
        bert_code_out=self.fc768_300(bert_code_out)
        code_add=torch.add(bert_code_out,codeOutMerge)
#         print(bert_report_out.shape,report_outMerge.shape,bert_code_out.shape,codeOutMerge.shape)
#         out=torch.cat([bert_report_out,report_outMerge,bert_code_out,codeOutMerge],dim=1)
        out=torch.cat([report_add,code_add],dim=1)
        # print('out shape :', out.shape)
        # out = out.unsqueeze(1)
        # print('out unsqueeze shape :', out.shape)
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print('out shape :', out.shape)
#         out = self.dropout(out)
        # print('out shape :', out.shape)
        out = self.fc(out)
        # print('最终的输出shape :',out.shape)
        return out
if __name__ == '__main__':
    model = TextCNNClassifer()
    ids=[]
    mask=[]
    token_type_ids=[]
    res=model(ids,mask,token_type_ids)
