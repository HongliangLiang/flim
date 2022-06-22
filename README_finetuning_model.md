lang=java

aspectj **************
python3 run.py     --output_dir=./saved_models/$lang     --config_name=microsoft/codetive.py aspectj/aspectj  aspectj > log_220120_aspectj_crossproject_1-19-37-38 2>&1 &bert-base     --model_name_or_path=microsoft/codebert-base     --tokenizer_name=microsoft/codebert-base     --do_train     --train_data_file=dataset/$lang/aspectj/train_cross.jsonl     --eval_data_file=dataset/$lang/aspectj/valid.jsonl     --test_data_file=dataset/$lang/aspectj/test.jsonl     --codebase_file=dataset/$lang/aspectj/codebase_cross.jsonl     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 32     --eval_batch_size 64     --learning_rate 2e-5     --seed 123456 2>&1| tee saved_models/$lang/aspectj/train.log

进入tracking的步骤，
nohup python3 cal_vec.py > 1018_log_tomcat_calvec_tar 2>&1 &
nohup python3 save_birt.py > 1018_log_tomcat_merge_feature_tar 2>&1 &
./load_data_to_joblib_memmap.py tomcat/tomcat
nohup python3 train_adaptive.py tomcat/tomcat  > 1018_log_tomcat_add_3738_tar_max_mean 2>&1 &

./load_data_to_joblib_memmap.py swt/swt
nohup python3 train_adaptive_feature_combine.py swt/swt > log_220117_swt_mix_4-37-38 2>&1 &

./load_data_to_joblib_memmap.py jdt/jdt
nohup python3 train_adaptive_feature_combine.py jdt/jdt > log_220117_jdt_mix_4-37-38 2>&1 &

./load_data_to_joblib_memmap.py eclipse_platform_ui/eclipse_platform_ui
nohup python3 train_adaptive_feature_combine.py eclipse_platform_ui/eclipse_platform_ui > log_220117_eclipse_platform_ui_mix_4-37-38 2>&1 &

./load_data_to_joblib_memmap.py birt/birt
nohup python3 train_adaptive_feature_combine.py birt/birt > log_220117_birt_mix_4-37-38 2>&1 &


# Code Search

## Data Preprocess

Different from the setting of [CodeSearchNet](husain2019codesearchnet), the answer of each query is retrieved from the whole development and testing code corpus instead of 1,000 candidate codes. Besides, we observe that some queries contain content unrelated to the code, such as a link ``http://..." that refers to external resources.  Therefore, we filter following examples to improve the quality of the dataset. 

- Remove comments in the code

- Remove examples that codes cannot be parsed into an abstract syntax tree.

- Remove examples that #tokens of documents is < 3 or >256

- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)

- Remove examples that documents are not English.

Data statistic about the cleaned dataset for code document generation is shown in this Table.

| PL         | Training |  Dev   |  Test  | Candidates code |
| :--------- | :------: | :----: | :----: | :-------------: |
| Python     | 251,820  | 13,914 | 14,918 |     43,827      |
| PHP        | 241,241  | 12,982 | 14,014 |     52,660      |
| Go         | 167,288  | 7,325  | 8,122  |     28,120      |
| Java       | 164,923  | 5,183  | 10,955 |     40,347      |
| JavaScript |  58,025  | 3,885  | 3,291  |     13,981      |
| Ruby       |  24,927  | 1,400  | 1,261  |      4,360      |

You can download and preprocess data using the following command.
```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```

## Dependency 

- pip install torch
- pip install transformers

## Fine-Tune

We fine-tuned the model on 2*V100-16G GPUs. 
```shell
lang=java
mkdir -p ./saved_models/$lang
python3 run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/train.log
```
## Inference and Evaluation

```shell
lang=java
python3 run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/test.log
```

## Demo

```shell
cd demo
python demo.py
```






0530
nohup python3 run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_3hard_1_train_data_less.txt \
--dev_file train_3hard_1_valid_data_less.txt \
--max_seq_length 256 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 3e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir /data/hdj/tracking_buggy_files/joblib_memmap_swt/data \
--output_dir ./models/java2  \
--file_prefix /data/hdj/tracking_buggy_files/swt/swt/ \
--model_name_or_path microsoft/codebert-base > log_0531_token_10method_less.txt 2>&1 &
