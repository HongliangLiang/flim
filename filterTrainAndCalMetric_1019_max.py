import pandas as pd
import numpy as np
import sys

def calculate_metrics(verification_df, k_range=range(1, 21)):
    average_precision_per_bug_report = []
    reciprocal_ranks = []
    # calculate per each query (bug report)
    accuracy_at_k = dict.fromkeys(k_range, 0)
    bug_report_number = 0
    for bug_report, bug_report_files_dataframe in verification_df.groupby(level=0, sort=False):
        min_fix_result = bug_report_files_dataframe[bug_report_files_dataframe['used_in_fix'] == 1.0]['result'].min()
        bug_report_files_dataframe2 = bug_report_files_dataframe[bug_report_files_dataframe["result"] >= min_fix_result]
        sorted_df = bug_report_files_dataframe2.sort_values(ascending=False, by=['result'])
        if sorted_df.shape[0] == 0:
            sorted_df = bug_report_files_dataframe.copy().sort_values(ascending=False, by=['result'])
            # print((bug_report_files_dataframe['used_in_fix'] == 1.0).sum())

        precision_at_k = []
        # precision per k in range
        tmp = sorted_df
        a = range(1, tmp.shape[0] + 1)
        tmp['position'] = pd.Series(a, index=tmp.index)

        large_k_p = tmp[(tmp['used_in_fix'] == 1.0)]['position'].tolist()
        unique_results = sorted_df['result'].unique().tolist()
        unique_results.sort()
        for fk in large_k_p:
            k = int(fk)
            k_largest = unique_results[-k:]

            largest_at_k = sorted_df[sorted_df['result'] >= min(k_largest)]
            real_fixes_at_k = (largest_at_k['used_in_fix'] == 1.0).sum()

            p = float(real_fixes_at_k) / float(k)
            precision_at_k.append(p)

        # average precision is sum of k precisions divided by K
        # K is set of positions of relevant documents in the ranked list
        average_precision = pd.Series(precision_at_k).mean()
        # average_precision = pd.Series(precision_at_k).sum() / float(large_k)
        average_precision_per_bug_report.append(average_precision)

        # accuracy
        for k in k_range:
            k_largest = unique_results[-k:]

            largest_at_k = sorted_df[sorted_df['result'] >= min(k_largest)]
            real_fixes_at_k = largest_at_k['used_in_fix'][(largest_at_k['used_in_fix'] == 1.0)].count()
            if real_fixes_at_k >= 1:
                accuracy_at_k[k] += 1

        # reciprocal rank
        indexes_of_fixes = np.flatnonzero(sorted_df['used_in_fix'] == 1.0)
        if indexes_of_fixes.size == 0:
            reciprocal_ranks.append(0.0)
        else:
            first_index = indexes_of_fixes[0]
            reciprocal_rank = 1.0 / (first_index + 1)
            reciprocal_ranks.append(reciprocal_rank)
        # bug number
        bug_report_number += 1

        del bug_report, bug_report_files_dataframe

    # print("average_precision_per_bug_report", average_precision_per_bug_report)
    mean_average_precision = pd.Series(average_precision_per_bug_report).mean()
    # print('mean average precision', mean_average_precision)
    mean_reciprocal_rank = pd.Series(reciprocal_ranks).mean()
    # print('mean reciprocal rank', mean_reciprocal_rank)
    for k in k_range:
        try:
            accuracy_at_k[k] = accuracy_at_k[k] / bug_report_number
        except:
            accuracy_at_k[k]=0
        # print('accuracy for k', accuracy_at_k[k], k)
    return accuracy_at_k, mean_average_precision, mean_reciprocal_rank

# project='tomcat'
# swt_all_results_df_after=pd.read_pickle('/data/hdj/tracking_buggy_files/joblib_memmap_'+project+'/all_results_df_after_1017.pickle')


# fold_number=8
#project='eclipse_platform_ui'
#fold_number=12
#project='birt'
#fold_number=8
# project='aspectj'
# fold_number=1
# project='tomcat'
# fold_number=2
project=sys.argv[1]
fold_number=int(sys.argv[2])
mod=sys.argv[3]
# swt_all_results_df_after=pd.read_pickle('/data/hdj/tracking_buggy_files/featureMerge/all_results_df_after_1017_notar'+mod+'_'+project+'.pickle')
# swt_all_results_df_after=pd.read_pickle('/data/hdj/tracking_buggy_files/featureMerge/all_results_df_after_1017_'+mod+'_'+project+'_both.pickle')
swt_all_results_df_after=pd.read_pickle('/data/hdj/tracking_buggy_files/featureMerge/all_results_df_after_1017_'+mod+'_'+project+'_max.pickle')
# swt_all_results_df_after=pd.read_pickle('/data/hdj/tracking_buggy_files/featureMerge/all_results_df_after_1017_'+mod+'_'+project+'_mean.pickle')


train_fold_03=[]
# project='jdt'
# fold_number=12
# project='swt'
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

all_data_len=len(bidList)
train_len=int(all_data_len*0.2)
val_len=train_len#+int(all_data_len*0.2)
# filterList=list(bidList)[:train_len]
filterList=list(bidList)[:val_len]
len(filterList),len(set(filterList))

filterSetList=list(set(filterList))
tar=set()
for id in bidList:
    if id not in filterSetList:
        tar.add(id)
existList=swt_all_results_df_after.index.get_level_values(0).unique()
finalTar=set()
for t in tar:
    if t in existList:
        finalTar.add(t)
print('剩余的report id',len(tar),'原始长度',len(set(bidList)),'训练长度',len(filterSetList),'final',len(finalTar))
swt_all_results_df_after=swt_all_results_df_after.loc[list(finalTar)]
print(calculate_metrics(swt_all_results_df_after))
