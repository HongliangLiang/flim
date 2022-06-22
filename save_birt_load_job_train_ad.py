# code_urls[:30]

import re
import pandas as pd
import numpy as np
from collections import defaultdict
#
import gc
# project='jdt'
# fold_number=12

# project='eclipse_platform_ui'
# fold_number=12
# project='tomcat'
# fold_number=2
# project='aspectj'
# fold_number=1
# project='swt'
# fold_number=8
# project='birt'
# fold_number=8
test_fold_03=[]
#TODO change
project='tomcat'
fold_number=2
tar=False#执行birt
top_k_method=5
# project='swt'
# fold_number=13


#TODO change
for k in range(fold_number+1):
    test_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_testing_fold_'+str(k)+'_raw')
    test_fold_03.append(test_fold_k)
    print(test_fold_k.shape)

test_fold_all=pd.concat(test_fold_03)
del test_fold_03
gc.collect()
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
#     if i==10:
#         break
#     print(len(train_set & test_set),len(train_set))
# tar=True #执行eclipse 执行tomcat

if tar:
    nl_vecs = np.load("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/nl_vecs_tar1018.npy")
    # nl_vecs[0]
    code_vecs=np.load("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/code_vecs_tar1018.npy")

    code_urls=np.load("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/code_urls_tar1018.npy")
    nl_urls=np.load("/data/hdj/tracking_buggy_files/joblib_memmap_"+project+"/nl_urls_tar1018.npy")
else:
    nl_vecs = np.load("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_vecs_cross_"+str(top_k_method)+".npy")
    # nl_vecs[0]
    code_vecs = np.load("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_vecs_cross_"+str(top_k_method)+".npy")

    code_urls = np.load("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/code_urls_cross_"+str(top_k_method)+".npy")
    nl_urls = np.load("/data/hdj/tracking_buggy_files/joblib_memmap_" + project + "/nl_urls_cross_"+str(top_k_method)+".npy")

reportId2nlvec=dict()
for reportId,nl_vec in zip(nl_urls,nl_vecs):
    reportId2nlvec[reportId]=nl_vec
codeId2codevec=defaultdict(list)
for codeId,code_vec in zip(code_urls,code_vecs):
    codeId2codevec[codeId].append(code_vec)
report2codeScores = []
print('计算分数')
for i, (reportId, codeId) in enumerate(reportId2codeId.items()):
    #     print(reportId,codeId)
    reportId = reportId.split('_')[0]
    nl_vec = reportId2nlvec[reportId]
    code_vec = codeId2codevec[codeId]

    code_vec = np.array(code_vec)
    #     print(code_vec)
    #             print(len(nl_vec))
    scores = np.matmul(nl_vec, code_vec.T)
    report2codeScores.append(list(scores))
# print(len(code_vec),scores)
#     if i==10:
#         break

bid_list=[]
codeId_list=[]
for key,val in reportId2codeId.items():
    try:
        bid_list.append(key.split('_')[0])
        codeId_list.append(val)
    except:
        print(key,val)
print('开始生成dataframe')
features=[[0]*19 for _ in range(len(bid_list))]
for i,val in enumerate(report2codeScores):
    for j,fea in enumerate(val):
        features[i][j]=fea
    features[i][-2]=max(val)
    features[i][-1]=np.mean(val)

# features
# features=[[0] for _ in range(len(bid_list))]
# for i,val in enumerate(report2codeScores):
#         features[i][0]=max(val)
#         features[i][0]=np.mean(val)
df = pd.DataFrame(features,columns=['f'+str(i) for i in range(20,39)])
# df = pd.DataFrame(features,columns=['f'+str(i) for i in range(20,21)])
min_df=pd.DataFrame(df.min()).transpose()
max_df=pd.DataFrame(df.max()).transpose()
df=(df - min_df.min()) / (max_df.max() - min_df.min())
del features
del code_vecs
del nl_vecs
gc.collect()
df['bid']=bid_list
df['fid']=codeId_list
test_fold_all.index.names=['bid','fid']
test_fold_all.head()
df.set_index(['bid','fid'], inplace = True)
# df.head()
# df.to_csv('/data/hdj/tracking_buggy_files/'+project+'/'+'tar1018_'+'df_features.csv',index=False)
print('开始merge')
# fold_number=1
# tar=True

if tar:
    for k in range(fold_number+1):
        training_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_training_fold_'+str(k)+'_raw')
        training_fold_k.index.names=['bid','fid']
    #     training_fold_k.head()
        all_dataframe=training_fold_k.join(df,how='inner')
        # all_dataframe.head()
        all_dataframe.to_pickle('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_normalized_training_fold_'+str(k)+'_tar1018')

    for k in range(fold_number + 1):
        training_fold_k = pd.read_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_testing_fold_' + str(k) + '_raw')
        training_fold_k.index.names = ['bid', 'fid']
        #     training_fold_k.head()
        all_dataframe = training_fold_k.join(df, how='inner')
        # all_dataframe.head()
        all_dataframe.to_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_testing_fold_' + str(k)+'_tar1018')
else:
    for k in range(fold_number + 1):
        training_fold_k = pd.read_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_training_fold_' + str(
                k) + '_raw')
        training_fold_k.index.names = ['bid', 'fid']
        #     training_fold_k.head()
        all_dataframe = training_fold_k.join(df, how='inner')
        # all_dataframe.head()
        all_dataframe.to_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_training_fold_' + str(
                k) + '_cross220116')

    for k in range(fold_number + 1):
        training_fold_k = pd.read_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_testing_fold_' + str(k) + '_raw')
        training_fold_k.index.names = ['bid', 'fid']
        #     training_fold_k.head()
        all_dataframe = training_fold_k.join(df, how='inner')
        # all_dataframe.head()
        all_dataframe.to_pickle(
            '/data/hdj/tracking_buggy_files/' + project + '/' + project + '_normalized_testing_fold_' + str(
                k) + '_cross220116')