
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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from collections import defaultdict
test_fold_03=[]
#TODO change
# project='tomcat'#准备执行tomcat
# project='birt'
import sys
project=sys.argv[1]
print(project)
#project='aspectj'
# project='jdt'
# project='swt'



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer, args):
    # code
    code = js['code_tokens']
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = js['docstring_tokens']
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]


class args(object):
    """A single set of features of data."""

    def __init__(self):
        self.code_length = 512
        self.nl_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 64
        self.learning_rate = 2e-5

        self.output_dir = '/data/hdj/data/CodeBERT/Siamese-model/saved_models/java'
        self.model_name_or_path = 'microsoft/codebert-base'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = args()
# project='eclipse_platform_ui'
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
query_file_name = '/data/hdj/tracking_buggy_files/joblib_memmap_' + project + '/report.jsonl'
args.code = False
query_dataset = TextDataset(tokenizer, args, query_file_name)
query_sampler = SequentialSampler(query_dataset)
query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=32)

args.code = True
code_file_name = '/data/hdj/tracking_buggy_files/joblib_memmap_' + project + '/code.jsonl'
code_dataset = TextDataset(tokenizer, args, code_file_name)
code_sampler = SequentialSampler(code_dataset)
code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=512, num_workers=32)

model = RobertaModel.from_pretrained(args.model_name_or_path)
model=Model(model)
# checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
checkpoint_prefix = 'checkpoint-best-mrr-new/model.bin'
# checkpoint_prefix = 'checkpoint-best-hdj/model.bin'
output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
print(output_dir)
model.load_state_dict(torch.load(output_dir),strict=False)
model.to(args.device)

model.eval()
code_vecs=[]
nl_vecs=[]
for batch in query_dataloader:
    nl_inputs = batch[1].to(args.device)
    with torch.no_grad():
        nl_vec = model(nl_inputs=nl_inputs)
        nl_vecs.append(nl_vec.cpu().numpy())

for batch in code_dataloader:
    code_inputs = batch[0].to(args.device)
    with torch.no_grad():
        code_vec= model(code_inputs=code_inputs)
        code_vecs.append(code_vec.cpu().numpy())
code_vecs = np.concatenate(code_vecs, 0)
nl_vecs = np.concatenate(nl_vecs, 0)
type(code_vecs), type(nl_vecs)
print(len(code_vecs), len(nl_vecs))
nl_urls = []
code_urls = []
for example in query_dataset.examples:
    nl_urls.append(example.url)

for example in code_dataset.examples:
    code_urls.append(example.url)
len(nl_urls), len(code_urls)
# tar=True #tomcat 执行了tar
tar=False#tomcat执行了mix 执行birt
top_k_method=5
if tar:
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/nl_vecs_tar1018_"+str(top_k_method)+".npy",nl_vecs)
    # b = np.load("filename.npy")
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/code_vecs_tar1018_"+str(top_k_method)+".npy",code_vecs)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/nl_urls_tar1018_"+str(top_k_method)+".npy",nl_urls)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/code_urls_tar1018_"+str(top_k_method)+".npy",code_urls)
else:
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_vecs_"+str(top_k_method)+"_MIX.npy", nl_vecs)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_vecs_"+str(top_k_method)+"_MIX.npy", code_vecs)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_urls_"+str(top_k_method)+"_MIX.npy", nl_urls)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_urls_"+str(top_k_method)+"_MIX.npy", code_urls)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_vecs_"+str(top_k_method)+"_CSN.npy", nl_vecs)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_vecs_"+str(top_k_method)+"_CSN.npy", code_vecs)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_urls_"+str(top_k_method)+"_CSN.npy", nl_urls)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_urls_"+str(top_k_method)+"_CSN.npy", code_urls)


    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_vecs_cross_"+str(top_k_method)+"_notar.npy", nl_vecs)
    # # b = np.load("filename.npy")
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_vecs_cross_"+str(top_k_method)+"_notar.npy", code_vecs)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_urls_cross_"+str(top_k_method)+"_notar.npy", nl_urls)
    # np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_urls_cross_"+str(top_k_method)+"_notar.npy", code_urls)


