import glob
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import sys, re
import os
import csv
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
from torch.nn import CrossEntropyLoss
import datetime
import os
import json
from unqlite import UnQLite
import string
import re
import inflection
import nltk
from nltk.stem.porter import PorterStemmer
from assets import stop_words, java_keywords
from parsers import Parser
import pandas as pd
from collections import namedtuple
# dataTest=pd.DataFrame()
# dataTest['id']=[1,2,3]
# dataTest['name']=['hdj','gzh','cxl']
# dataTest.to_pickle('/data/hdj/cross_project_trans/report/aspectj_test_middle/student.pkl')

def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?;\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

#
# def load_bin_vec(fname, vocab):
#     word_vecs = {}
#     model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)#模型的导入
#     for word in vocab:
#         if word in model.vocab:
#             word_vecs[word] = model[word]
#     return word_vecs
#
#
# def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
#     """
#     For words that occur in at least min_df documents, create a separate word vector.
#     0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
#     """
#     # for word in vocab:
#     #     if word not in word_vecs and vocab[word] >= min_df:
#     #         word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
#     i = 0  # 统计多少不在预训练词表里
#     for word in vocab:
#         if word not in word_vecs and vocab[word] >= min_df:
#             print('不在词表里词示例 ', word)
#             i += 1
#             word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
#     print('一共 ', i, ' 个词不在词表里')

def load_bin_vec(google,hdj, vocab,hdjWord2Vec):
    word_vecs = {}
    if hdjWord2Vec:
        w2v = Word2Vec.load(hdj)#模型的导入
    else:
        w2v=gensim.models.KeyedVectors.load_word2vec_format(hdj, binary=True)
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(hdj, binary=True)#模型的导入
    vocab_emd=w2v.wv.vocab
    print('max_token : ', (w2v.wv.syn0.shape[0]))
    # hdj_vocab = w2v.wv.vocab
    # max_token = w2v.wv.syn0.shape[0]  # 字典的单词总量
    # print('max_token :',max_token)
    # goo_model = gensim.models.KeyedVectors.load_word2vec_format(google, binary=True)#模型的导入
    for word in vocab:
        if word in vocab_emd:
            word_vecs[word] = w2v[word]
        # else:
            # print(word,' not in 词向量')
    print('load_bin_vec :',len(word_vecs))
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=2, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    # for word in vocab:
    #     if word not in word_vecs and vocab[word] >= min_df:
    #         word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    i = 0  # 统计多少不在预训练词表里
    print(type(word_vecs),type(vocab))
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # print('不在词表里词示例 ', word)
            i += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    print('一共 ', i, ' 个词不在词表里')
def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word,emb in word_vecs.items():
        # print(type(word),word,emb)
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    print('get W ',len(W),len(word_idx_map))
    return W, word_idx_map

def getIdxfrom_sent(sent, word_idx_map, code_maxk):
    x = []
    #    pad = filter_h - 1
    #    for i in xrange(pad):
    #        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    if len(x) <= code_maxk:
        while len(x) < code_maxk:
            x.append(0)
    if len(x) >= code_maxk:
        while len(x) > code_maxk:
            x.pop()
    return x


def getIdxfrom_sent_n(sent, max_l, word_idx_map, filter_h=5):
    #TODO 为什么先pad 0 后再pad 0 ?
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def get_W(word_vecs, k=300):
#     vocab_size = len(word_vecs)
#     word_idx_map = dict()
#     W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
#     W[0] = np.zeros(k, dtype='float32')
#     i = 1
#     for word in word_vecs:
#         W[i] = word_vecs[word]
#         word_idx_map[word] = i
#         i += 1
#     return W, word_idx_map


def alignData(data, code_maxl,code_maxk):
    try:
        mm = data.shape[0]
        nn = data.shape[1]
    except:
        tt = code_maxl
        nn=code_maxk
        return np.zeros(nn * tt, dtype="int").reshape(tt, nn)
    if mm < code_maxl:
        tt = code_maxl - mm
        aa = np.zeros(nn * tt, dtype="int").reshape(tt, nn)
        new_data = np.vstack((data, aa))
    else:
        new_data = data[:code_maxl, :]
    return new_data

def random_permutation(train_report, train_code, labels):
    labels = np.asarray(labels)
    labels = labels.reshape(len(labels))
    datasets_size = len(labels)
    labels_rand = np.zeros(datasets_size, dtype="int")
    zz = np.random.permutation(np.arange(datasets_size))
    train_report_rand, train_code_rand = [], []
    for i in range(datasets_size):
        labels_rand[i] = labels[zz[i]]
        train_report_rand.append(train_report[zz[i]])
        train_code_rand.append(train_code[zz[i]])
    return train_report_rand, train_code_rand, labels_rand

def getReport(bug_id,report_data, maxlen,tag):
    # print(type(bug_id),type(report_data['bug_id'][0]))
    summary=None
    description=None
    # print('hhh: ',bug_id,tag)
    try:
        description=eval(report_data[report_data['bug_id'] == (bug_id)]['pos_tagged_descriptions'].values[0])['stemmed']#['unstemmed']
        # print(description)
        summary=eval(report_data[report_data['bug_id'] == (bug_id)]['pos_tagged_summarys'].values[0])['stemmed']#['unstemmed']
    except Exception as e:
        print('except : ',bug_id)
    # print(summary)
    report=None
    if description!=None and summary!=None:
        summary.extend(description)
        if len(summary) > maxlen:
            cut_words = []
            for i in range(maxlen):
                cut_words.append(summary[i])
            report = " ".join(cut_words)
        else:
            report = " ".join(summary)
    return report
class ReportPreprocessing:
    """Class to preprocess bug reports"""

    __slots__ = ['bug_reports']

    def __init__(self, bug_reports):
        self.bug_reports = bug_reports

    def pos_tagging(self):
        """Extracing specific pos tags from bug reports' summary and description"""

        for report in self.bug_reports.values():
            # Tokenizing using word_tokeize for more accurate pos-tagging
            summ_tok = nltk.word_tokenize(report.summary)
            try:
                desc_tok = nltk.word_tokenize(report.description)
            except Exception as e:
                print('出问题了 ', report.description)
            sum_pos = nltk.pos_tag(summ_tok)
            desc_pos = nltk.pos_tag(desc_tok)

            report.pos_tagged_summary = [token for token, pos in sum_pos
                                         if 'NN' in pos or 'VB' in pos]
            report.pos_tagged_description = [token for token, pos in desc_pos
                                             if 'NN' in pos or 'VB' in pos]

    def tokenize(self):
        """Tokenizing bug reports into tokens"""

        for report in self.bug_reports.values():
            report.summary = nltk.wordpunct_tokenize(report.summary)
            report.description = nltk.wordpunct_tokenize(report.description)

    def _split_camelcase(self, tokens):

        # Copy tokens
        returning_tokens = tokens[:]

        for token in tokens:
            split_tokens = re.split(fr'[{string.punctuation}]+', token)

            # If token is split into some other tokens
            if len(split_tokens) > 1:
                returning_tokens.remove(token)
                # Camel case detection for new tokens
                for st in split_tokens:
                    camel_split = inflection.underscore(st).split('_')
                    if len(camel_split) > 1:
                        returning_tokens.append(st)
                        returning_tokens += camel_split
                    else:
                        returning_tokens.append(st)
            else:
                camel_split = inflection.underscore(token).split('_')
                if len(camel_split) > 1:
                    returning_tokens += camel_split

        return returning_tokens

    def split_camelcase(self):
        """Split CamelCase identifiers"""

        for report in self.bug_reports.values():
            report.summary = self._split_camelcase(report.summary)
            report.description = self._split_camelcase(report.description)
            report.pos_tagged_summary = self._split_camelcase(report.pos_tagged_summary)
            report.pos_tagged_description = self._split_camelcase(report.pos_tagged_description)

    def normalize(self):
        """Removing punctuation, numbers and also lowercase conversion"""

        # Building a translate table for punctuation and number removal
        punctnum_table = str.maketrans({c: None for c in string.punctuation + string.digits})

        for report in self.bug_reports.values():
            summary_punctnum_rem = [token.translate(punctnum_table)
                                    for token in report.summary]
            desc_punctnum_rem = [token.translate(punctnum_table)
                                 for token in report.description]
            pos_sum_punctnum_rem = [token.translate(punctnum_table)
                                    for token in report.pos_tagged_summary]
            pos_desc_punctnum_rem = [token.translate(punctnum_table)
                                     for token in report.pos_tagged_description]

            report.summary = [token.lower() for token
                              in summary_punctnum_rem if token]
            report.description = [token.lower() for token
                                  in desc_punctnum_rem if token]
            report.pos_tagged_summary = [token.lower() for token
                                         in pos_sum_punctnum_rem if token]
            report.pos_tagged_description = [token.lower() for token
                                             in pos_desc_punctnum_rem if token]

    def remove_stopwords(self):
        """Removing stop words from tokens"""

        for report in self.bug_reports.values():
            report.summary = [token for token in report.summary
                              if token not in stop_words]
            report.description = [token for token in report.description
                                  if token not in stop_words]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary
                                         if token not in stop_words]
            report.pos_tagged_description = [token for token in report.pos_tagged_description
                                             if token not in stop_words]

    def remove_java_keywords(self):
        """Removing Java language keywords from tokens"""

        for report in self.bug_reports.values():
            report.summary = [token for token in report.summary
                              if token not in java_keywords]
            report.description = [token for token in report.description
                                  if token not in java_keywords]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary
                                         if token not in java_keywords]
            report.pos_tagged_description = [token for token in report.pos_tagged_description
                                             if token not in java_keywords]

    def stem(self):
        """Stemming tokens"""

        # Stemmer instance
        stemmer = PorterStemmer()

        for report in self.bug_reports.values():
            report.summary = dict(zip(['stemmed', 'unstemmed'],
                                      [[stemmer.stem(token) for token in report.summary],
                                       report.summary]))

            report.description = dict(zip(['stemmed', 'unstemmed'],
                                          [[stemmer.stem(token) for token in report.description],
                                           report.description]))

            report.pos_tagged_summary = dict(zip(['stemmed', 'unstemmed'],
                                                 [[stemmer.stem(token) for token
                                                   in report.pos_tagged_summary],
                                                  report.pos_tagged_summary]))

            report.pos_tagged_description = dict(zip(['stemmed', 'unstemmed'],
                                                     [[stemmer.stem(token) for token
                                                       in report.pos_tagged_description],
                                                      report.pos_tagged_description]))

    def preprocess(self):
        """Run the preprocessing"""

        self.pos_tagging()
        self.tokenize()
        self.split_camelcase()
        self.normalize()
        self.remove_stopwords()
        self.remove_java_keywords()
        self.stem()


class BugReport:
    """Class representing each bug report"""

    __slots__ = ['summary', 'description',
                 'pos_tagged_summary', 'pos_tagged_description']

    def __init__(self, summary, description):
        self.summary = summary
        self.description = description
        self.pos_tagged_summary = None
        self.pos_tagged_description = None


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?;\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_string_code(string):
    m = re.compile(r'/\*.*?\*/', re.S)
    outstring = re.sub(m, '', string)
#     m = re.compile(r'package*', re.S)
#     outstring = re.sub(m, '', outstring)
#     m = re.compile(r'import*', re.S)
#     outstring = re.sub(m, '', outstring)
    m = re.compile(r'//.*')
    outtmp = re.sub(m, '', outstring)
class SrcPreprocessing_fulltext:
    def __init__(self,code_dir):
        self.src_files=self.get_repo_source_files(code_dir)

    def tokenize(self):
        """Tokenizing source codes into tokens"""
        for key,vals in self.src_files.items():
            tmp=[]
            for val in vals:
                tmp.append(nltk.wordpunct_tokenize(val))
            self.src_files[key]= tmp

    def _split_camelcase(self,tokens):

        # Copy tokens
        returning_tokens = tokens[:]

        for token in tokens:
            split_tokens = re.split(fr'[{string.punctuation}]+', token)

            # If token is split into some other tokens
            if len(split_tokens) > 1:
                returning_tokens.remove(token)
                # Camel case detection for new tokens
                for st in split_tokens:
                    camel_split = inflection.underscore(st).split('_')
                    if len(camel_split) > 1:
                        returning_tokens.append(st)
                        returning_tokens += camel_split
                    else:
                        returning_tokens.append(st)
            else:
                camel_split = inflection.underscore(token).split('_')
                if len(camel_split) > 1:
                    returning_tokens += camel_split

        return returning_tokens

    def split_camelcase(self):
        """Split CamelCase identifiers"""

        for key,val_list in self.src_files.items():
            tmp = []
            # print('vals',val_list)
            for vals in val_list:
                # print('vals',vals)
                tmp.append(self._split_camelcase(vals))
            self.src_files[key] = tmp
    def normalize(self):
        """Removing punctuation, numbers and also lowercase conversion"""

        # Building a translate table for punctuation and number removal
        punctnum_table = str.maketrans({c: None for c in string.punctuation + string.digits})

        for key,vals in self.src_files.items():
            tmp = []
            for val in vals:
                # print(val)
                content_punctnum_rem = [token.translate(punctnum_table)for token in val]
                tmp.append([token.lower() for token in content_punctnum_rem if token])
            self.src_files[key] = tmp
    def remove_stopwords(self):
        """Removing stop words from tokens"""

        for key, val_list in self.src_files.items():
            tmp=[]
            for val in val_list:
                after_removed=[token for token in val if token not in stop_words]
                if len(after_removed)>0:
                    tmp.append(after_removed)
            self.src_files[key] = tmp
    def remove_java_keywords(self):
        """Removing Java language keywords from tokens"""

        for key, val_list in self.src_files.items():
            tmp=[]
            for val in val_list:
                after_removed =[token for token in val  if token not in java_keywords]
                if len(after_removed) > 0:
                    tmp.append(after_removed)
            self.src_files[key] = tmp

    def stem(self):
        """Stemming tokens"""

        # Stemmer instance
        stemmer = PorterStemmer()

        for key, val_list in self.src_files.items():
            tmp = []
            for val in val_list:
                tmp.append(dict(zip(['stemmed', 'unstemmed'],[[stemmer.stem(token) for token in val], val])))
            self.src_files[key] = tmp
    def preprocess(self):
        """Run the preprocessing"""

        # self.pos_tagging()
        print('tokenize')
        self.tokenize()
        print('split_camelcase')
        self.split_camelcase()
        print('normalize')
        self.normalize()
#         print('remove_stopwords')
#         self.remove_stopwords()
#         print('remove_java_keywords')
#         self.remove_java_keywords()
        print('stem')
        self.stem()
    def get_repo_source_files(self,code_dir):
        files = dict()
        start_dir = os.path.normpath(code_dir)  # 规范path字符串形式
        for dir_, dir_names, file_names in os.walk(start_dir):  # os.walk()方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
#             for filename in [f for f in file_names if f.endswith(".java")]:
            for i,filename in enumerate(file_names) :
#                 if i==10:
#                     break
                src_name = os.path.join(dir_, filename)  # 所有java文件名
                with open(src_name, encoding='utf-8', mode='r') as src_file:
                    lines = src_file.readlines()
                file_key = src_name.split(start_dir)[1]
                file_key = file_key[len(os.sep):]  # os.sep 不用考虑是linux 的‘/’ 或者是windows的 ‘\’

                # 过滤掉 copyright
                new_lines = []
                beforeTag = True
                for line in lines:
                    line = line.lstrip().replace('\n', '')
                    if 'package' in line or 'import' in line or 'class' in line:
                        beforeTag = False
                    if beforeTag == False:
                        new_lines.append(line)

                # 分离code和description
                source = []
                # if file_key == 'org.aspectj\modules\\ajde\\testdata\examples\coverage\ModelCoverage.java':
                #     print('newline: ',new_lines)
                for line in new_lines:
                    # if file_key == 'org.aspectj\modules\\ajde\\testdata\examples\coverage\ModelCoverage.java':
                    #     print(line)
                    if line == "": continue
                    if line.startswith("@"): continue
                    if line.find('/*') >= 0: continue
                    if line.startswith('*'): continue
                    if line.find('*/') >= 0: continue
                    # if line.find('import')>=0:continue
                    if line.startswith('//'):
                        continue
                    else:
                        if line.find('//') >= 0:
                            index = line.index('//')
                            line = line[:index]
                        # source += (line + '\n')
                        source.append(line + '\n')

                # print(source,'\n******************************\n')
                # print(file_key)
                # if file_key=='org.aspectj\modules\\ajde\\testdata\examples\coverage\ModelCoverage.java':
                #     print('here ',file_key,source)
                files[file_key] = source
        return  files

srcs = {
    '/data/hdj/cross_project_trans/report/bert/swt_code': '/data/hdj/cross_project_trans/report/bert/swt_code_clean'}

for src, clean_path in srcs.items():
    src_addresses = glob.glob(src + '/*', recursive=True)
    code_prep = SrcPreprocessing_fulltext(src)
    code_prep.preprocess()
    print('总共java文件数 ，处理后java文件数', len(src_addresses), len(code_prep.src_files))
    j = 0
    for path, val_list in code_prep.src_files.items():

        #         path=pathNorm(path)
        #         path=os.path.join(clean_path[i],path)
        path = os.path.join(clean_path, path)
        #         print(path)
        #         mkdir(path)
        #         if len(val_list)==0:
        #             print(path,' is null')
        #             j+=1
        #             continue
        with open(path, 'w', encoding='utf-8') as f_out:
            for val in val_list:
                f_out.write(' '.join(val['stemmed']) + '\n')
                #     print(reg,'共有',j,'个文件被过滤掉')
    i += 1