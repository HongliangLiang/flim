#! /bin/bash
python3 cal_vec.py eclipse_platform_ui && python3 save_birt.py eclipse_platform_ui 12 &&  python3 load_data_to_joblib_memmap.py eclipse_platform_ui/eclipse_platform_ui && nohup python3 train_adaptive.py eclipse_platform_ui/eclipse_platform_ui  > log_220117_eclipse_platform_ui_crossproject_1-19-37-38 2>&1 &