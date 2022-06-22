
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
project='tomcat'#准备执行tomcat
# project='birt'
# project='aspectj'
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
query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

args.code = True
code_file_name = '/data/hdj/tracking_buggy_files/joblib_memmap_' + project + '/code.jsonl'
code_dataset = TextDataset(tokenizer, args, code_file_name)
code_sampler = SequentialSampler(code_dataset)
code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=512, num_workers=6)

model = RobertaModel.from_pretrained(args.model_name_or_path)
model=Model(model)
checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
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
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_vecs_cross_"+str(top_k_method)+".npy", nl_vecs)
    # b = np.load("filename.npy")
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_vecs_cross_"+str(top_k_method)+".npy", code_vecs)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_urls_cross_"+str(top_k_method)+".npy", nl_urls)
    np.save("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_urls_cross_"+str(top_k_method)+".npy", code_urls)






# fold_number=2
# project='eclipse_platform_ui' 已经执行过了

# project='tomcat'
# fold_number=2
# project='aspectj'
# fold_number=1
# project='swt'
# fold_number=8
# project='birt'
# fold_number=8
#TODO change
# for k in range(fold_number+1):
#     test_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_testing_fold_'+str(k)+'_raw')
#     test_fold_03.append(test_fold_k)
#     print(test_fold_k.shape)
# test_fold_all=pd.concat(test_fold_03)
# print(test_fold_all.shape)
# test_fold_all.head()
# # training_fold_10.head()
# report_idx=test_fold_all.index.get_level_values(0).unique()
# code_idx=test_fold_all.index.get_level_values(1).unique()
# len(report_idx),len(code_idx)
#
# reportId2codeId=dict()
# for i,report in enumerate(report_idx):
#     test_set=test_fold_all.loc[report].index.get_level_values(0).unique()
# #     print(len(test_set))
#     for codeId in test_set:
#         reportId2codeId[report+'_'+codeId]=codeId
# #获取bug_report TODO
# import json
# def load_bug_reports(bug_report_file_path):
#     """load bug report file (the one generated from xml)"""
#     with open(bug_report_file_path) as bug_report_file:
#         bug_reports = json.load(bug_report_file)
#         return bug_reports
# # bug_report_file_path='/data/hdj/tracking_buggy_files/'+swt+'.json'
# # bug_report_file_path='/data/hdj/tracking_buggy_files/eclipse/eclipse_platform_ui.json'
# # bug_report_file_path='/data/hdj/tracking_buggy_files/aspectj/aspectj.json'
# bug_report_file_path='/data/hdj/tracking_buggy_files/'+project+'/'+project+'.json'
# bug_reports = load_bug_reports(bug_report_file_path)
# #这个是原始的report ，我还可以获取经过分词的report
# # bug_report_id='a694733'
# # current_bug_report = bug_reports[bug_report_id]['bug_report']
# # current_bug_report
#
# ast_cache_collection_db=UnQLite("/data/hdj/tracking_buggy_files/"+project+"/"+project+"_ast_cache_collection_db",
#                                              flags=0x00000100 | 0x00000001)
#
# bid_list = list(report_idx)
# print(len(bid_list))
# summarys = []
# descriptions = []
# for bug_report_id in bid_list:
#     current_bug_report = bug_reports[bug_report_id]['bug_report']
#     summarys.append(current_bug_report['summary'])
#     descriptions.append(current_bug_report['description'])
#
# report_dataFrame = pd.DataFrame({'summary': summarys, 'description': descriptions}, index=bid_list)
# report_dataFrame.index.names = ['bid']
#
#
# def remove_twoHeadWord(string):
#     contents = string.split(' ')[2:]
#     return ' '.join(contents)
#
#
# report_dataFrame['summary'] = report_dataFrame['summary'].apply(remove_twoHeadWord)
# report_dataFrame.fillna('', inplace=True)
# report_dataFrame.head()
#
# def clean_string_report(string):
#     outtmp=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
# #     outtmp=[token for token in outtmp if token not in stop_words]
# #     outtmp= [token for token in outtmp if token not in java_keywords]
#     outtmp=[token for token in outtmp if token.isalnum() ]
#     return ' '.join(outtmp)
# with open('/data/hdj/tracking_buggy_files/joblib_memmap_'+project+'/report.jsonl','w',encoding='utf-8') as f_out:
#        for row in report_dataFrame.iterrows():
# #             print(row)
#             task1=dict()
#             task1['url']=row[0]
#             task1['docstring_tokens']=clean_string_report(row[1].summary+' '+row[1].description)
#             task1['code_tokens']=''
#             out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#             f_out.write(out_line)
#
# def clean_code_names(ast_file):
#     names=ast_file['classNames']+ast_file['superclassNames']+ast_file['interfaceNames']+ast_file['methodNames']#+ast_file['variableNames']
#     names=[' '.join(name.split('.')) for name in names]
# #     print(names)
#     return ' '.join(names)
# def camel_case_split(identifier):
#     matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
#     return [m.group(0) for m in matches]
# def mergeTokenziedSource(tokenizedSource):#dict 类型
#     return ' '.join(tokenizedSource.keys())
# removed = u'!"#%&\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
# utf_translate_table = dict((ord(char), u' ') for char in removed)
# stop_words = set(stopwords.words('english'))
#
# def tokenize(text, stemmer):
#     sanitized_text = text.translate(utf_translate_table)
#     tokens = wordpunct_tokenize(sanitized_text)
#     all_tokens = []
#     for token in tokens:
#         additional_tokens = camel_case_split(token)
#         if len(additional_tokens)>1:
#             for additional_token in additional_tokens:
#                 all_tokens.append(additional_token)
#         all_tokens.append(token)
#     return Counter([stemmer.stem(token) for token in all_tokens if token.lower() not in stop_words])
# def convert_dict2string_set(dict_list):
#     counter_list = set()
#     for dict_item in dict_list:
#         for k,v in dict_item.items():
#             if len(k)<=2:
#                 continue
#             counter_list.add(k)
# #     print('len dict_set :',len(dict_list),'merge len :',len(counter_list))
#     return ' '.join(counter_list)
# def convert_dict2string_list(dict_list):
#     counter_list = []
#     for dict_item in dict_list:
#         for k,v in dict_item.items():
#             counter_list.append(k)
# #     print('len dict_list :',len(dict_list),'merge len :',len(counter_list))
#     return ' '.join(counter_list)
# def clean_string_code(string):
#     m = re.compile(r'/\*.*?\*/', re.S)
#     outstring = re.sub(m, '', string)
#     m = re.compile(r'package*', re.S)
#     outstring = re.sub(m, '', outstring)
#     m = re.compile(r'import*', re.S)
#     outstring = re.sub(m, '', outstring)
#     m = re.compile(r'//.*')
#     outtmp = re.sub(m, '', outstring)
#     for char in ['\r\n', '\r', '\n']:
#         outtmp = outtmp.replace(char, ' ')
#     outtmp=' '.join(outtmp.split())
#     outtmp=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',outtmp)
#     return ' '.join(outtmp)
# def get_method_top_k(choosed_methods:list,k=50):
#     method_list=[]
#     for string_bef in choosed_methods:
# #         print('string_bef :',string_bef)
#         string=clean_string_code(string_bef)
#         if '{'  in string and '}'in string :
#             method_list.append(string)
# #     print('函数体的方法数量: ',len(method_list))
#     num=len(method_list)
#     method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
#     if len(method_list)==0:
#         for string_bef in choosed_methods:
# #         print('string_bef :',string_bef)
#             string=clean_string(string_bef)
#             method_list.append(string)
#         method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
#         num=len(method_list)
#         return method_list[:k],num
# #     print(len(method_list))
# #     for method in method_list:
# #         print(method)
#     return method_list[:k],num
# # method_top_k,num=get_method_top_k(swt_ast_file['methodContent'],10)
# # num
# def clean_string(string):
#     outstring_list=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
#     return ' '.join(outstring_list)
#
#
# all_ast_index = list(code_idx)
# print(type(all_ast_index))
#
# all_ast_file_tokenizedMethods = []
# all_ast_file_tokenizedClassNames = []
# all_ast_file_tokenizedMethodNames = []
# all_ast_file_tokenizedVariableNames = []
# all_ast_file_tokenizedComments = []
#
# all_ast_file_source = []
# all_ast_file_names = []
# all_ast_file_methods = []
# for ast_index in all_ast_index:
#     ast_file = pickle.loads(ast_cache_collection_db[ast_index])
#
#     all_ast_file_tokenizedMethods.append(convert_dict2string_set(ast_file['tokenizedMethods']))
#     all_ast_file_tokenizedClassNames.append(convert_dict2string_set(ast_file['tokenizedClassNames']))
#     all_ast_file_tokenizedMethodNames.append(convert_dict2string_set(ast_file['tokenizedMethodNames']))
#     all_ast_file_tokenizedVariableNames.append(convert_dict2string_set(ast_file['tokenizedVariableNames']))
#     all_ast_file_tokenizedComments.append(convert_dict2string_set(ast_file['tokenizedComments']))
#
#     all_ast_file_source.append(clean_string_code(ast_file['rawSourceContent']))
#     all_ast_file_names.append(clean_code_names(ast_file))
#     top_k_method, num = get_method_top_k(ast_file['methodContent'], 10)
#     print('total method num :', len(ast_file['methodContent']), ' 有方法体的函数数量 :', num, ' choosed method num :',
#           len(top_k_method))
#     all_ast_file_methods.append(top_k_method)
#
# all_ast_index_dataframe = pd.DataFrame(
#     {'all_ast_file_methods': all_ast_file_methods, 'tokenizedMethods': all_ast_file_tokenizedMethods,
#      'tokenizedClassNames': all_ast_file_tokenizedClassNames, 'tokenizedMethodNames': all_ast_file_tokenizedMethodNames,
#      'tokenizedVariableNames': all_ast_file_tokenizedVariableNames, 'tokenizedComments': all_ast_file_tokenizedComments,
#      'source': all_ast_file_source, 'names': all_ast_file_names}, index=all_ast_index)
# # all_ast_index_dataframe = pd.DataFrame({'all_ast_file_methods':all_ast_file_methods, 'source':all_ast_file_source,'names':all_ast_file_names},index=all_ast_index)
#
# with open('/data/hdj/tracking_buggy_files/joblib_memmap_' + project + '/code.jsonl', 'w', encoding='utf-8') as f_out:
#     for row in all_ast_index_dataframe.iterrows():
#         #             print(row)
#         all_methods = row[1].all_ast_file_methods
#         for method in all_methods:
#             task1 = dict()
#             task1['url'] = row[0]
#             task1['docstring_tokens'] = ''
#             task1['code_tokens'] = method
#             out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#             f_out.write(out_line)
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].names
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].source
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedMethods开始
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].tokenizedMethods
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedMethods结束
#         # 写tokenizedClassNames开始
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].tokenizedClassNames
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedClassNames结束
#         # 写tokenizedMethodNames开始
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].tokenizedMethodNames
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedMethodNames结束
#         # 写tokenizedVariableNames开始
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].tokenizedVariableNames
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedVariableNames结束
#
#         # 写tokenizedComments开始
#         task1 = dict()
#         task1['url'] = row[0]
#         task1['docstring_tokens'] = ''
#         task1['code_tokens'] = row[1].tokenizedComments
#         out_line = json.dumps(task1, ensure_ascii=False) + '\n'
#         f_out.write(out_line)
#         # 写tokenizedComments结束