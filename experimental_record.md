# tracking_buggy_files
This repository contains scripts to process two datasets, feature preparation code and implementation of algorithms from publication "Tracking Buggy Files: New Efficient Adaptive Bug Localization Algorithm".
Main directory contains python code to prepare features and conduct experiments.
The java-ast-extractor directory contains 4 programs enriching source code files with ast trees, utilized during feature construction.
The ast trees are stored as git notes per each source file.
The java 8 and apache maven are required to compile java-ast-extractor.
Rest of scripts require python 3 and python 2.

# How to replicate dataset - example for AspectJ project, using already existing git notes
* Download original dataset
* Clone repository https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/
* Fetch git notes containing ast trees and import graphs
```
git fetch origin refs/notes/commits:refs/notes/commits
git fetch origin refs/notes/tokenized_counters:refs/notes/tokenized_counters
git fetch origin refs/notes/graph:refs/notes/graph
```
* Convert project files from xml to json:
```
./process_bug_reports.py AspectJ.xml ../tracking_buggy_files_aspectj_dataset/ aspectj_base.json
./fix_and_augment.py aspectj_base.json ../tracking_buggy_files_aspectj_dataset/ > aspectj_aug.json
./pick_bug_freq.py aspectj_aug.json ../tracking_buggy_files_aspectj_dataset/ > aspectj.json
```
  * To add missing bug reports descriptions adjust results of "process_bug_reports" using additional script, and use result json file in in the next step:
```
./add_missing_description_as_separate_reports.py aspectj_base.json aspectj_base_with_descriptions.json BUGZILLA_API_KEY BUGZILLA_API_URL
```
* Calculate features - result files will be stored using prefix "aspectj":
```
./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_eclipse_platform_ui_dataset /data/hdj/tracking_buggy_files/eclipse_platform_ui/eclipse_platform_ui.json eclipse_platform_ui/eclipse_platform_ui

./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_swt_dataset /data/hdj/tracking_buggy_files/swt.json swt/swt

./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_aspectj_dataset /data/hdj/tracking_buggy_files/aspectj/aspectj.json aspectj/aspectj

./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_birt_dataset /data/hdj/tracking_buggy_files/birt/birt.json birt/birt

./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_tomcat_dataset /data/hdj/tracking_buggy_files/tomcat/tomcat.json tomcat/tomcat

./create_ast_cache.py /data/hdj/SourceFile/tracking_buggy_files_jdt_dataset /data/hdj/tracking_buggy_files/jdt/jdt.json jdt/jdt

./vectorize_ast.py aspectj.json aspectj

./vectorize_enriched_api.py aspectj.json aspectj
./convert_tf_idf.py aspectj.json aspectj
./calculate_feature_3.py aspectj.json aspectj
./retrieve_features_5_6.py aspectj.json aspectj
./calculate_notes_graph_features.py aspectj.json aspectj ../tracking_buggy_files_aspectj_dataset/
./calculate_vectorized_features.py aspectj.json aspectj
./save_normalized_fold_dataframes.py aspectj.json aspectj
```
# How to replicate adaptive method results
Example for AspectJ project, using same data prefix as feature calculation
```
./load_data_to_joblib_memmap.py aspectj
./load_data_to_joblib_memmap.py eclipse_platform_ui/eclipse_platform_ui
./load_data_to_joblib_memmap.py aspectj/aspectj

./load_data_to_joblib_memmap.py aspectj/aspectj
./train_adaptive.py eclipse_platform_ui/eclipse_platform_ui
nohup python3 train_adaptive.py aspectj/aspectj > 0622_log_apectj_add_3738_nofine 2>&1 &

./load_data_to_joblib_memmap.py tomcat/tomcat
nohup python3 train_adaptive.py tomcat/tomcat> 0619_log_tomcat_add_3738 2>&1 &

./load_data_to_joblib_memmap.py swt/swt
./train_adaptive.py swt/swt
nohup python3 train_adaptive.py swt/swt > 0619_log_swt_add_32-38 2>&1 &

./load_data_to_joblib_memmap.py birt/birt
./train_adaptive.py birt/birt
nohup python3  train_adaptive.py birt/birt > 0620_log_birt_rem151719_add3738_all 2>&1 &
nohup python3  train_adaptive.py birt/birt > 1019_log_birt_all_max_mean 2>&1 &

./load_data_to_joblib_memmap.py jdt/jdt
nohup python3 train_adaptive.py jdt/jdt > 1019_log_jdt_all_mean 2>&1 &
```

find . | grep  _files$ | xargs rm
find . | grep  .npz | xargs rm
find . | grep  .png | xargs rm
find . | grep  tomcat_wrong_sort_ | xargs rm
find . | grep  swt_wrong_sort_ | xargs rm
find . | grep  jdt_wrong_sort_ | xargs rm
find . | grep  birt_wrong_sort_ | xargs rm
find . | grep  eclipse_platform_ui_wrong_sort_ | xargs rm
training_fold_k= pd.read_pickle('/data/hdj/tracking_buggy_files/'+'aspectj'+'/'+'aspectj'+'_normalized_training_fold_'+str(1)+'_raw')
nohup python3 cal_vec.py > 0622_log_tomcat_calvec_nofine 2>&1 &
nohup python3 save_birt.py > 0623_log_tomcat_merge_feature 2>&1 &
./load_data_to_joblib_memmap.py tomcat/tomcat
nohup python3 train_adaptive.py tomcat/tomcat  > 0623_log_tomcat_add_3738_nofine 2>&1 &

nohup python3 cal_vec.py > 0622_log_swt_calvec_nofine 2>&1 &
nohup python3 save_birt.py > 0623_log_swt_merge_feature 2>&1 &
nohup python3 train_adaptive.py swt/swt  > 0623_log_swt_add_3738_nofine 2>&1 &


211019******
birt TAR
nohup python3 cal_vec.py > 1018_log_birt_calvec_tar 2>&1 &
nohup python3 save_birt.py > 1018_log_birt_merge_feature_tar 2>&1 &
./load_data_to_joblib_memmap.py birt/birt
nohup python3 train_adaptive.py birt/birt  > 1018_log_birt_add_3738_tar_max_mean 2>&1 &


birt MIX
nohup python3 cal_vec.py > 1020_log_birt_calvec_mix 2>&1 &
nohup python3 save_birt.py > 1020_log_birt_merge_feature_mix 2>&1 &
./load_data_to_joblib_memmap.py birt/birt
nohup python3 train_adaptive.py birt/birt  > 1020_log_birt_mix_max_mean 2>&1 &
