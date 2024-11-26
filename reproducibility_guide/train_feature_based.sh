#!/bin/bash

# Commands to train all feature based models for all sizes, save the trained models, and evaluate them and save their results

# Nearest Neighbors
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# Linear SVM
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# Decision Tree
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# Random Forest
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# Neural Net
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# AdaBoost
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# Naive Bayes
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true

# QDA
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_catch22.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_catch22_normalized/ #--eval-true


# Nearest Neighbors
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=knn --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# Linear SVM
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=svc_linear --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# Decision Tree
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=decision_tree --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# Random Forest
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=random_forest --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# Neural Net
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=mlp --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# AdaBoost
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=ada_boost --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# Naive Bayes
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=bayes --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true

# QDA
python3.8 train_feature_based.py --path=data/normalized/OBSEA_4/TSFRESH_OBSEA_4_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_8/TSFRESH_OBSEA_8_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_16/TSFRESH_OBSEA_16_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_32/TSFRESH_OBSEA_32_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_64/TSFRESH_OBSEA_64_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_128/TSFRESH_OBSEA_128_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_256/TSFRESH_OBSEA_256_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_512/TSFRESH_OBSEA_512_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_768/TSFRESH_OBSEA_768_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
python3.8 train_feature_based.py --path=data/normalized/OBSEA_1024/TSFRESH_OBSEA_1024_custom0.25.npy --classifier=qda --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/weights_custom0.25_normalized/  #--eval-true
