1. fine-tune CodeBERT
please download from this link and copy to CodeBERT-finetune/dataset/java 
https://drive.google.com/drive/folders/1ezG7gNKw4Z0IxmWBfAqOHHRGftMsBCas?usp=sharing

take tomcat for example **************
python3 run.py     --output_dir=./saved_models/java     --config_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base     --tokenizer_name=microsoft/codebert-base     --do_train     --train_data_file=dataset/java/tomcat/train_cross.jsonl     --eval_data_file=dataset/java/tomcat/valid.jsonl     --test_data_file=dataset/java/tomcat/test.jsonl     --codebase_file=dataset/java/tomcat/codebase_cross.jsonl     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 32     --eval_batch_size 64     --learning_rate 2e-5     --seed 123456 
2. train LTR model
please refer to https://github.com/mfejzer/tracking_buggy_files download all the data

python3 /data/hdj/tracking_buggy_files/generate_function_and_text.py tomcat 2 && python3 /data/hdj/tracking_buggy_files/cal_vec.py tomcat && python3 /data/hdj/tracking_buggy_files/save_birt.py tomcat 2 &&  python3 /data/hdj/tracking_buggy_files/load_data_to_joblib_memmap.py tomcat/tomcat && nohup python3 /data/hdj/tracking_buggy_files/train_adaptive.py tomcat/tomcat  > log_tomcat_crossproject 2>&1 &
