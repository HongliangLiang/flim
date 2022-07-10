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
# fold_number=2
# project='eclipse_platform_ui'

# project='tomcat'
# fold_number=2
# project='aspectj'
# fold_number=1
# project='swt'
# fold_number=8
# project='birt'
# fold_number=8
import sys
project=sys.argv[1]
fold_number=sys.args[2]
print(project)

TODO change
for k in range(fold_number+1):
    test_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_testing_fold_'+str(k)+'_raw')
    test_fold_03.append(test_fold_k)
    print(test_fold_k.shape)
test_fold_all=pd.concat(test_fold_03)
print(test_fold_all.shape)
test_fold_all.head()
# training_fold_10.head()
report_idx=test_fold_all.index.get_level_values(0).unique()
code_idx=test_fold_all.index.get_level_values(1).unique()
len(report_idx),len(code_idx)

reportId2codeId=dict()
for i,report in enumerate(report_idx):
    test_set=test_fold_all.loc[report].index.get_level_values(0).unique()
#     print(len(test_set))
    for codeId in test_set:
        reportId2codeId[report+'_'+codeId]=codeId
#获取bug_report TODO
import json
def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports
# bug_report_file_path='/data/hdj/tracking_buggy_files/'+swt+'.json'
# bug_report_file_path='/data/hdj/tracking_buggy_files/eclipse/eclipse_platform_ui.json'
# bug_report_file_path='/data/hdj/tracking_buggy_files/aspectj/aspectj.json'
bug_report_file_path='/data/hdj/tracking_buggy_files/'+project+'/'+project+'.json'
bug_reports = load_bug_reports(bug_report_file_path)
#这个是原始的report ，我还可以获取经过分词的report
# bug_report_id='a694733'
# current_bug_report = bug_reports[bug_report_id]['bug_report']
# current_bug_report

ast_cache_collection_db=UnQLite("/data/hdj/tracking_buggy_files/"+project+"/"+project+"_ast_cache_collection_db",
                                             flags=0x00000100 | 0x00000001)

bid_list = list(report_idx)
print(len(bid_list))
summarys = []
descriptions = []
for bug_report_id in bid_list:
    current_bug_report = bug_reports[bug_report_id]['bug_report']
    summarys.append(current_bug_report['summary'])
    descriptions.append(current_bug_report['description'])

report_dataFrame = pd.DataFrame({'summary': summarys, 'description': descriptions}, index=bid_list)
report_dataFrame.index.names = ['bid']


def remove_twoHeadWord(string):
    contents = string.split(' ')[2:]
    return ' '.join(contents)


report_dataFrame['summary'] = report_dataFrame['summary'].apply(remove_twoHeadWord)
report_dataFrame.fillna('', inplace=True)
report_dataFrame.head()

def clean_string_report(string):
    outtmp=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
#     outtmp=[token for token in outtmp if token not in stop_words]
#     outtmp= [token for token in outtmp if token not in java_keywords]
    outtmp=[token for token in outtmp if token.isalnum() ]
    return ' '.join(outtmp)
with open('/data/hdj/tracking_buggy_files/joblib_memmap_'+project+'/report.jsonl','w',encoding='utf-8') as f_out:
       for row in report_dataFrame.iterrows():
#             print(row)
            task1=dict()
            task1['url']=row[0]
            task1['docstring_tokens']=clean_string_report(row[1].summary+' '+row[1].description)
            task1['code_tokens']=''
            out_line = json.dumps(task1, ensure_ascii=False) + '\n'
            f_out.write(out_line)

def clean_code_names(ast_file):
    names=ast_file['classNames']+ast_file['superclassNames']+ast_file['interfaceNames']+ast_file['methodNames']#+ast_file['variableNames']
    names=[' '.join(name.split('.')) for name in names]
#     print(names)
    return ' '.join(names)
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]
def mergeTokenziedSource(tokenizedSource):#dict 类型
    return ' '.join(tokenizedSource.keys())
removed = u'!"#%&\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
utf_translate_table = dict((ord(char), u' ') for char in removed)
stop_words = set(stopwords.words('english'))

def tokenize(text, stemmer):
    sanitized_text = text.translate(utf_translate_table)
    tokens = wordpunct_tokenize(sanitized_text)
    all_tokens = []
    for token in tokens:
        additional_tokens = camel_case_split(token)
        if len(additional_tokens)>1:
            for additional_token in additional_tokens:
                all_tokens.append(additional_token)
        all_tokens.append(token)
    return Counter([stemmer.stem(token) for token in all_tokens if token.lower() not in stop_words])
def convert_dict2string_set(dict_list):
    counter_list = set()
    for dict_item in dict_list:
        for k,v in dict_item.items():
            if len(k)<=2:
                continue
            counter_list.add(k)
#     print('len dict_set :',len(dict_list),'merge len :',len(counter_list))
    return ' '.join(counter_list)
def convert_dict2string_list(dict_list):
    counter_list = []
    for dict_item in dict_list:
        for k,v in dict_item.items():
            counter_list.append(k)
#     print('len dict_list :',len(dict_list),'merge len :',len(counter_list))
    return ' '.join(counter_list)
def clean_string_code(string):
    m = re.compile(r'/\*.*?\*/', re.S)
    outstring = re.sub(m, '', string)
    m = re.compile(r'package*', re.S)
    outstring = re.sub(m, '', outstring)
    m = re.compile(r'import*', re.S)
    outstring = re.sub(m, '', outstring)
    m = re.compile(r'//.*')
    outtmp = re.sub(m, '', outstring)
    for char in ['\r\n', '\r', '\n']:
        outtmp = outtmp.replace(char, ' ')
    outtmp=' '.join(outtmp.split())
    outtmp=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',outtmp)
    return ' '.join(outtmp)
def get_method_top_k(choosed_methods:list,k=50):
    method_list=[]
    for string_bef in choosed_methods:
#         print('string_bef :',string_bef)
        string=clean_string_code(string_bef)
        if '{'  in string and '}'in string :
            method_list.append(string)
#     print('函数体的方法数量: ',len(method_list))
    num=len(method_list)
    method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
    if len(method_list)==0:
        for string_bef in choosed_methods:
#         print('string_bef :',string_bef)
            string=clean_string(string_bef)
            method_list.append(string)
        method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
        num=len(method_list)
        return method_list[:k],num
#     print(len(method_list))
#     for method in method_list:
#         print(method)
    return method_list[:k],num
# method_top_k,num=get_method_top_k(swt_ast_file['methodContent'],10)
# num
def clean_string(string):
    outstring_list=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
    return ' '.join(outstring_list)


all_ast_index = list(code_idx)
print(type(all_ast_index))

all_ast_file_tokenizedMethods = []
all_ast_file_tokenizedClassNames = []
all_ast_file_tokenizedMethodNames = []
all_ast_file_tokenizedVariableNames = []
all_ast_file_tokenizedComments = []

all_ast_file_source = []
all_ast_file_names = []
all_ast_file_methods = []
for ast_index in all_ast_index:
    ast_file = pickle.loads(ast_cache_collection_db[ast_index])

    all_ast_file_tokenizedMethods.append(convert_dict2string_set(ast_file['tokenizedMethods']))
    all_ast_file_tokenizedClassNames.append(convert_dict2string_set(ast_file['tokenizedClassNames']))
    all_ast_file_tokenizedMethodNames.append(convert_dict2string_set(ast_file['tokenizedMethodNames']))
    all_ast_file_tokenizedVariableNames.append(convert_dict2string_set(ast_file['tokenizedVariableNames']))
    all_ast_file_tokenizedComments.append(convert_dict2string_set(ast_file['tokenizedComments']))

    all_ast_file_source.append(clean_string_code(ast_file['rawSourceContent']))
    all_ast_file_names.append(clean_code_names(ast_file))
    top_k_method, num = get_method_top_k(ast_file['methodContent'], 10)
    print('total method num :', len(ast_file['methodContent']), ' 有方法体的函数数量 :', num, ' choosed method num :',
          len(top_k_method))
    all_ast_file_methods.append(top_k_method)

all_ast_index_dataframe = pd.DataFrame(
    {'all_ast_file_methods': all_ast_file_methods, 'tokenizedMethods': all_ast_file_tokenizedMethods,
     'tokenizedClassNames': all_ast_file_tokenizedClassNames, 'tokenizedMethodNames': all_ast_file_tokenizedMethodNames,
     'tokenizedVariableNames': all_ast_file_tokenizedVariableNames, 'tokenizedComments': all_ast_file_tokenizedComments,
     'source': all_ast_file_source, 'names': all_ast_file_names}, index=all_ast_index)
# all_ast_index_dataframe = pd.DataFrame({'all_ast_file_methods':all_ast_file_methods, 'source':all_ast_file_source,'names':all_ast_file_names},index=all_ast_index)

with open('/data/hdj/tracking_buggy_files/joblib_memmap_' + project + '/code.jsonl', 'w', encoding='utf-8') as f_out:
    for row in all_ast_index_dataframe.iterrows():
        #             print(row)
        all_methods = row[1].all_ast_file_methods
        for method in all_methods:
            task1 = dict()
            task1['url'] = row[0]
            task1['docstring_tokens'] = ''
            task1['code_tokens'] = method
            out_line = json.dumps(task1, ensure_ascii=False) + '\n'
            f_out.write(out_line)
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].names
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].source
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedMethods开始
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].tokenizedMethods
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedMethods结束
        # 写tokenizedClassNames开始
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].tokenizedClassNames
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedClassNames结束
        # 写tokenizedMethodNames开始
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].tokenizedMethodNames
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedMethodNames结束
        # 写tokenizedVariableNames开始
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].tokenizedVariableNames
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedVariableNames结束

        # 写tokenizedComments开始
        task1 = dict()
        task1['url'] = row[0]
        task1['docstring_tokens'] = ''
        task1['code_tokens'] = row[1].tokenizedComments
        out_line = json.dumps(task1, ensure_ascii=False) + '\n'
        f_out.write(out_line)
        # 写tokenizedComments结束