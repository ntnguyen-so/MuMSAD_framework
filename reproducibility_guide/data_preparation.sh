#!/bin/bash

# Compute all Oracles scores and save them in ‘data/TSB/metrics/<randomness>_ORACLE-<acc>’
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=true
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=lucky
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=unlucky
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-3
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-4
# python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-5

# Compute Averaging Ensemble scores and save them in ‘data/TSB/metrics/AVG_ENS’
# python3 run_avg_ens.py --n_jobs=12

# Compute the windowed datasets and save them in 'data/TSB_<window_size>' 
# python3 create_windows_dataset.py --save_dir=data/ --path=data/TSB/data/ --metric_path=data/TSB/metrics/ --window_size=all --metric=AUC_PR

# Compute the features per dataset and save them in 'data/TSB_<window_size>/'TSFRESH_TSB_<window_size>.csv', super intentinsive process requires 100s GBs of RAM







# python3.8 generate_features.py --path=data/normalized/OBSEA_1024/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_768/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_512/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_256/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_128/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_64/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_32/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_16/
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_16/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_32/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_64/
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_128/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_256/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_512/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_768/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_1024/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_4/ 
# python3.8 generate_features.py --path=data/not_normalized/OBSEA_8/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_4/ 
# python3.8 generate_features.py --path=data/normalized/OBSEA_8/ 
# bash reproducibility_guide/train_feature_based.sh &
# bash reproducibility_guide/train_feature_based_not_normalized.sh
python3.8 run_workflow.py 0  &
python3.8 run_workflow.py 1  &
python3.8 run_workflow.py 2  &
python3.8 run_workflow.py 3  &
python3.8 run_workflow.py 4  &
python3.8 run_workflow.py 5  &
python3.8 run_workflow.py 6  &
python3.8 run_workflow.py 7  &
python3.8 run_workflow.py 8  &
python3.8 run_workflow.py 9  &
python3.8 run_workflow.py 10 &
python3.8 run_workflow.py 11 &
python3.8 run_workflow.py 12 &
python3.8 run_workflow.py 13 &
python3.8 run_workflow.py 14 &
python3.8 run_workflow.py 15 &
python3.8 run_workflow.py 16 &
python3.8 run_workflow.py 17 &
python3.8 run_workflow.py 18 &
python3.8 run_workflow.py 19 &
python3.8 run_workflow.py 20 &

# python3 generate_features.py --path=data/TSB_16/ ; python3.6 generate_features.py --path=data/TSB_32/ ; python3.6 generate_features.py --path=data/TSB_64/ ; python3.6 generate_features.py --path=data/TSB_128/ ; python3.6 generate_features.py --path=data/TSB_256/
