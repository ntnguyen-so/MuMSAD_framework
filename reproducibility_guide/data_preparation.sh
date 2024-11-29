#!/bin/bash

# If you want to create window data for a dataset called OBSEA and all window sizes from 4, 8, 16, ..., 1024. The metric to used is INTERPRETABILITY.
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=all --metric=INTERPRETABILITY --data_normalization=True

# In case you want to create window data for a specific window size
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=4 --metric=INTERPRETABILITY 
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=8 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=16 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=32 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=64 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=128 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=256 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=512 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=768 --metric=INTERPRETABILITY
python3 create_windows_dataset.py --name=OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=1024 --metric=INTERPRETABILITY
