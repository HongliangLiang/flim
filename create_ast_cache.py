#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <repository_path> <bug_reports.json> <data_prefix>
"""

import datetime
import json
import pickle
import subprocess
import sys

from joblib import Parallel, delayed

from date_utils import convert_commit_date
from multiprocessing import Pool
from operator import itemgetter
from timeit import default_timer
from tqdm import tqdm
from unqlite import UnQLite


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    repository_path = sys.argv[1]
    print("repository path", repository_path)
    bug_report_file_path = sys.argv[2]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[3]
    print("data prefix", data_prefix)

    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        process(bug_reports, repository_path, data_prefix)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def process(bug_reports, repository_path, data_prefix):
    #关键函数 准备ast解析出来的信息
    ast_cache = prepare_ast_cache(repository_path)
    #关键函数
    ast_cache_collection_db = UnQLite(data_prefix + "_ast_cache_collection_db")

    before = default_timer()
    # 将ast解析的信息进行保存
    for k, v in ast_cache.items():
        ast_cache_collection_db[k] = pickle.dumps(v, -1)
    after = default_timer()
    total = after - before
    print("total ast cache saving time ", total)
    # 保存完成

    # 关键函数 准备bug_report的信息
    # bug_report_files = prepare_bug_report_files(repository_path, bug_reports, ast_cache)
    # # 关键函数
    # before = default_timer()
    # #将bug_report信息进行保存
    # bug_report_files_collection_db = UnQLite(data_prefix + "_bug_report_files_collection_db")
    # for k, v in bug_report_files.items():
    #     bug_report_files_collection_db[k] = pickle.dumps(v, -1)
    #保存完成
    after = default_timer()
    total = after - before
    print("total bug report files saving time ", total)


def list_notes(repository_path, refs='refs/notes/commits'):
    #当ref=='refs/notes/tokenized_counters'时，解析出来的就是tokenized后的文件
    #git -C /data/hdj/SourceFile/tracking_buggy_files_tomcat_dataset notes --ref refs/notes/commits list
    #返回一堆 shas
    cmd = ' '.join(['git', '-C', repository_path, 'notes', '--ref', refs, 'list'])
    notes_lines = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read().decode('latin-1').split('\n')
    return notes_lines


def cat_file_blob(repository_path, sha, encoding='latin-1'):
    #git -C /data/hdj/SourceFile/tracking_buggy_files_tomcat_dataset cat-file blob ea6b85284ac8a5b67421bafebe99a7cc2ca6d73e > note_content.txt
    #返回所有文件相关的信息 包括分割好的方法和源码 dict格式
    cmd = ' '.join(['git', '-C', repository_path, 'cat-file', 'blob', sha])
    cat_file_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result = cat_file_process.stdout.read().decode(encoding)
    return result


def ls_tree(repository_path, sha):
    cmd = ' '.join(['git', '-C', repository_path, 'ls-tree', '-r', sha])
    ls_results = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read().decode('latin-1').split('\n')
    return ls_results[:-1]


def _process_notes(note,note_not_tokenize, repository_path):
    #ea6b85284ac8a5b67421bafebe99a7cc2ca6d73e 00001f23f4c6321b7b4468d2eae6b4a1e0e24bce
    (note_content_sha, note_object_sha) = note.split(' ')
    (note_content_sha_not_tokenize, note_object_sha_not_tokenize) = note_not_tokenize.split(' ')

    # note_content = cat_file_blob(repository_path, note_content_sha)
    note_content_not_tokenize = cat_file_blob(repository_path, note_content_sha_not_tokenize)
    # ast_extraction_result = json.loads(note_content)
    ast_extraction_result_not_tokenize = json.loads(note_content_not_tokenize)
    # ast_extraction_result['methodContent']=ast_extraction_result_not_tokenize['methodContent']
    # ast_extraction_result['rawSourceContent']=ast_extraction_result_not_tokenize['rawSourceContent']
    # ast_extraction_result['commentContent']=ast_extraction_result_not_tokenize['commentContent']
    # ast_extraction_result['methodNames']=ast_extraction_result_not_tokenize['methodNames']

    # return note_object_sha, ast_extraction_result
    # note_content_not_tokenize = cat_file_blob(repository_path, note_content_sha_not_tokenize)
    # ast_extraction_result_not_tokenize = json.loads(note_content_not_tokenize)
    return note_object_sha, ast_extraction_result_not_tokenize


def _f(args):
    return _process_notes(args[0], args[1],args[2])


def prepare_ast_cache(repository_path):
    tokenized_refs = 'refs/notes/tokenized_counters'
    ast_notes = list_notes(repository_path, refs=tokenized_refs)
    ast_notes_not_tokenize=list_notes(repository_path)
    # ast_notes_not_tokenize = [0 for _ in range(len(ast_notes))]
    print("existing tokenized notes ", len(ast_notes))
    print("existing not tokenized notes ", len(ast_notes_not_tokenize))

    before = default_timer()

    work = []
    for note,note_not_tokenize in zip(ast_notes,ast_notes_not_tokenize):
        if note != '' and note_not_tokenize!='':
            work.append((note,note_not_tokenize, repository_path))
    pool = Pool(12, maxtasksperchild=1)
    ast_cache = dict(tqdm(pool.imap(_f, work), total=len(work)))
    #dict({'shas':{'method':[],''}})
    #    r = Parallel(n_jobs=6*12)(delayed(__process_notes)(i, repository_path) for i in tqdm(ast_notes))
    #    ast_cache = dict(r)

    after = default_timer()
    total = after - before

    print("total ast cache retrieval time ", total)
    print("size of ast cache ", sys.getsizeof(ast_cache))
    return ast_cache


def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    for index, commit in enumerate(tqdm(bug_reports)):
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ', '').strip()
        commit_date = convert_commit_date(
            bug_reports[commit]['commit']['metadata']['date'].replace('Date:', '').strip())
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits


def _load_parent_commit_files(repository_path, commit, ast_cache):
    #找到该report提交时之前一个版本的所有文件
    parent = commit + '^'

    class_name_to_sha = {}
    sha_to_file_name = {}
    shas = []
    for ls_entry in ls_tree(repository_path, parent):
        #100644 blob fcb5575f03f63dda66c685269e619d9c2cf8be8b	java/javax/servlet/ServletContextAttributeListener.java
        (file_sha_part, file_name) = ls_entry.split('\t')
        file_sha = file_sha_part.split(' ')[2]
        # file_sha = intern(file_sha)
        # file_name = intern(file_name)
        if file_name.endswith(".java") and file_sha in ast_cache:
            # shas.append(intern(file_sha))
            file_sha_ascii = file_sha
            shas.append(file_sha_ascii)
            class_names = ast_cache[file_sha]['classNames']
            #这里应该是内部类 将该bug_report对应的父版本所有的java文件及其shas对应关系保存起来
            for class_name in class_names:
                class_name_ascii = class_name
                class_name_to_sha[class_name_ascii] = file_sha_ascii
            sha_to_file_name[file_sha_ascii] = file_name

    f_lookup = {'shas': shas, 'class_name_to_sha': class_name_to_sha, 'sha_to_file_name': sha_to_file_name}
    return commit.encode('ascii', 'ignore'), f_lookup


def prepare_bug_report_files(repository_path, bug_reports, ast_cache):
    sorted_commits = sort_bug_reports_by_commit_date(bug_reports)

    before = default_timer()

    r = Parallel(n_jobs=6 * 12, backend="threading")(
        delayed(_load_parent_commit_files)(repository_path, commit, ast_cache) for commit in tqdm(sorted_commits))
    bug_report_files = dict(r)

    after = default_timer()
    total = after - before
    print("total bug report files retrieval time ", total)
    #等于这里就是
    return bug_report_files


if __name__ == '__main__':
    main()
