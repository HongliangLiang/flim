import re
import pandas as pd
import nltk
import string
import glob
import string
import inflection
import json
import subprocess
from unqlite import UnQLite
import pickle
from scipy import sparse
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from re import finditer
import torch
from tqdm import tqdm, trange
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import transformers
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn import metrics
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
import os
from collections import defaultdict
from sklearn.metrics import f1_score
# from utils import (compute_metrics, convert_examples_to_features,
#                        output_modes, processors)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from collections import defaultdict
import json
project='tomcat'
fold_number=2
use_k=1
# project='eclipse_platform_ui'
# project='aspectj'
# project='birt'
# project='jdt'
# project='swt'
file_path='/data/hdj/tracking_buggy_files/joblib_memmap_'+project+'/'
report_file_path=file_path+'report_'+project+'.jsonl'
code_file_path=file_path+'code_'+project+'.jsonl'
reportData=defaultdict(list)
codeData=defaultdict(list)
with open(report_file_path) as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        reportData[js['url']].append(js)
with open(code_file_path) as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        codeData[js['url']].append(js)
train_fold_03=[]
# fold_number=8
# fold_number=12


# fold_number=1
# fold_number=12
# fold_number=8
for k in range(fold_number+1):
    test_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_training_fold_'+str(k)+'_raw')
    train_fold_03.append(test_fold_k)
    print(test_fold_k.shape)
train_fold_all=pd.concat(train_fold_03)
print(train_fold_all.shape)
train_fold_all.head()
train_all_data=train_fold_all[train_fold_all['used_in_fix']==1]
bidList = train_all_data.index.get_level_values(0)
fidList = train_all_data.index.get_level_values(1)
train_all_data.shape,len(bidList),len(fidList)

outPath='/data/hdj/data/CodeBERT/Siamese-model/dataset/java/'

trainOut=open(os.path.join(outPath,project+'/train_'+str(use_k)+'.jsonl'),'w',encoding='utf-8')
valOut=open(os.path.join(outPath,project+'/valid_'+str(use_k)+'.jsonl'),'w',encoding='utf-8')
testOut=open(os.path.join(outPath,project+'/test_'+str(use_k)+'.jsonl'),'w',encoding='utf-8')
codeBaseOut=open(os.path.join(outPath,project+'/codebase_'+str(use_k)+'.jsonl'),'w',encoding='utf-8')
all_data_len=len(bidList)
train_len=int(all_data_len*0.2)
val_len=train_len+int(all_data_len*0.2)
test_len=val_len+int(all_data_len*0.2)
for i,(bid, fid) in enumerate(zip(bidList,fidList)):
    print(i,bid,fid)
    report_data=reportData[bid]
    code_data=codeData[fid]
#     print(report_data)
#     print(code_data)
    if(i<=train_len):#train数据
        if use_k!=1:
            use= min(use_k,len(code_data))
            for i in range(use):
                tmp=dict()
                tmp['code']=code_data[i]['code_tokens']
                tmp['code_tokens']=code_data[i]['code_tokens'].split(' ')
                tmp['docstring']=report_data[0]['docstring_tokens']
                tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
                tmp['url']=bid+'_'+fid
                outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
                trainOut.write(outTmp)
        else:
            tmp=dict()
            tmp['code']=code_data[0]['code_tokens']
            tmp['code_tokens']=code_data[0]['code_tokens'].split(' ')
            tmp['docstring']=report_data[0]['docstring_tokens']
            tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
            tmp['url']=bid+'_'+fid
            outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
            trainOut.write(outTmp)
    elif(i<=val_len):#val数据
        if use_k!=1:
            use= min(use_k,len(code_data))
            for i in range(use):
                tmp=dict()
                tmp['code']=code_data[i]['code_tokens']
                tmp['code_tokens']=code_data[i]['code_tokens'].split(' ')
                tmp['docstring']=report_data[0]['docstring_tokens']
                tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
                tmp['url']=bid+'_'+fid
                outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
                valOut.write(outTmp)         
        else:
            tmp=dict()
            tmp['code']=''
            tmp['code_tokens']=''
            tmp['docstring']=report_data[0]['docstring_tokens']
            tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
            tmp['url']=bid+'_'+fid
            outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
            valOut.write(outTmp)
    elif(i<=test_len):#测试数据
        if use_k!=1:
            use= min(use_k,len(code_data))
            for i in range(use):
                tmp=dict()
                tmp['code']=code_data[i]['code_tokens']
                tmp['code_tokens']=code_data[i]['code_tokens'].split(' ')
                tmp['docstring']=report_data[0]['docstring_tokens']
                tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
                tmp['url']=bid+'_'+fid
                outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
                testOut.write(outTmp)
        else:
            tmp=dict()
            tmp['code']=''
            tmp['code_tokens']=''
            tmp['docstring']=report_data[0]['docstring_tokens']
            tmp['docstring_tokens']=report_data[0]['docstring_tokens'].split(' ')
            tmp['url']=bid+'_'+fid
            outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
            testOut.write(outTmp)
    tmp=dict()
    tmp['code']=code_data[0]['code_tokens']
    tmp['code_tokens']=code_data[0]['code_tokens'].split(' ')
    tmp['docstring']=''
    tmp['docstring_tokens']=''
    tmp['url']=bid+'_'+fid
    outTmp=json.dumps(tmp, ensure_ascii=False)+'\n'
    codeBaseOut.write(outTmp)
#     break
trainOut.close()
valOut.close()
testOut.close()
codeBaseOut.close()