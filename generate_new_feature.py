
test_fold_03=[]
#TODO change
# project='tomcat'
# fold_number=2
# project='swt'
# fold_number=13

project='eclipse_platform_ui'
fold_number=12
#TODO change
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