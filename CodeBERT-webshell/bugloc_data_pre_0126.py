import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
import csv
import gensim

'''
训练集
#1.读取/data/hdj/cross_project_trans/report/aspectj_80_training_pos.csv ,aspectj_80_training_neg.csv
数据格式 report_id,code_path,label: id,path,label

1)根据report_id去/data/hdj/cross_project_trans/report/aspectj_report_clean.csv 里获取description的清洗后的文本query
2)根据code_path去清洗后的代码文件中获取代码code
3)组成 label<CODESPLIT>report_id<CODESPLIT>code_path<CODESPLIT>query<CODESPLIT>code

验证集
同上
/data/hdj/cross_project_trans/report/aspectj_20_training_pos.csv,aspectj_20_training_neg.csv

'''



def getReport(bug_id,report_data):
    # print(type(bug_id),type(report_data['bug_id'][0]))
    summary=""
    # description=None
    # print('hhh: ',bug_id,tag)
    try:
        # description=eval(report_data[report_data['bug_id'] == int(bug_id)]['descriptions'].values[0])['unstemmed']#['unstemmed']
        # print(description)
        summary=eval(report_data[report_data['bug_id'] == int(bug_id)]['summarys'].values[0])['unstemmed']#['unstemmed']
    except Exception as e:
        print('except : ',bug_id)
    # print(summary)

    return  " ".join(summary)
def code_pre(outstring):
    m = re.compile(r'/\*.*?\*/', re.S)
    outtmp = re.sub(m, '', outstring)
    outstring = outtmp
    m = re.compile(r'//.*')
    outtmp = re.sub(m, '', outstring)
    outstring = outtmp
    m = re.compile(r'#.*')
    outtmp = re.sub(m, ' ', outstring)
    outstring = outtmp
    for char in ['\r\n', '\r', '\n']:
        outstring = outstring.replace(char, ' ')
    outstring = ' '.join(outstring.split())
    return outstring
def get_example(spamreader,examples,report_data,clean_path):
    i = 0
    for row in spamreader:
        if i == 0:
            i += 1
            continue
        # bug_report_name = "JDT_CNN/BugCorpus/BugCorpus/" + row[0].strip()
        # 根据bugid获得对应report
        bug_report_id = row[0].strip()
        bug_description = getReport(bug_report_id, report_data)

        source_code_name = row[1].strip()
        code_path = clean_path + source_code_name
        try:
            rf = open(code_path, 'r', encoding='utf-8', errors='ignore')
            data = rf.read()
        finally:
            # print(data)
            rf.close()
        code = code_pre(data)[:10000]

        label = row[2].strip()
        tmp = []
        tmp.append(label)
        tmp.append(bug_report_id)  # TODO 记录report_id
        tmp.append(source_code_name)
        tmp.append(bug_description)
        tmp.append(code)
        examples.append("<CODESPLIT>".join(tmp))
    return examples

def load_data( report_path, project_name, more_project_name, clean_path,tag):
    # labels_list, report_ids_list, path_list,report_list,code_list = [], [], [],[],[]
    examples=[]
    report_data = pd.read_csv(report_path + project_name + 'report_clean.csv')
    #读取pos的数据
    pos_spamreader = csv.reader(open(report_path + more_project_name + "training_pos.csv", newline=''))
    examples=get_example(pos_spamreader,examples,report_data,clean_path)
    #读取neg的数据
    neg_spamreader = csv.reader(open(report_path + more_project_name + "training_neg.csv", newline=''))
    examples = get_example(neg_spamreader, examples, report_data, clean_path)
    #aspectj_train.txt

    with open('/data/hdj/data/CodeBERT/data/codesearch/train_valid/'+project_name+tag+'.txt','w',encoding='utf-8') as f_out:
        for example in examples:
            f_out.write(example+'\n')
# code_maxl max statements per file

if __name__=="__main__":
    # test2()
    # test_bug_id()
    # project_names=['aspectj_','swt_','zxing_']
    report_path = '/data/hdj/cross_project_trans/report/'
    project_names=['aspectj_']
    code_clean = ['/home/hdj/bug_localization/data/AspectJ/AspectJ-1.5/clean/',]
                 # '/home/hdj/bug_localization/data/SWT/clean/', '/home/hdj/bug_localization/data/ZXing/clean/']
    i = 0
    for project_name in project_names:
        load_data( report_path,project_name, project_name + '80_', code_clean[i],'train')
        i += 1
        # pickle.dump([train_report, train_code, train_labels, W],open("parameters.in", "wb"))
        # pickle.dump([train_report, train_code, train_labels, W],open("JDT_CNN/parameters.in", "wb"))
        print("train Finish processing!")
    i = 0
    for project_name in project_names:
        load_data( report_path,project_name, project_name + '20_', code_clean[i],'val')
        i += 1
        # pickle.dump([train_report, train_code, train_labels, W],open("parameters.in", "wb"))
        # pickle.dump([train_report, train_code, train_labels, W],open("JDT_CNN/parameters.in", "wb"))
        print("val Finish processing!")