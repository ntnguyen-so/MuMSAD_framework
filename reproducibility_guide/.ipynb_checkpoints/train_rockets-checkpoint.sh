#!/bin/bash

python3 train_rocket.py --path=data/OBSEA_16/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_32/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_64/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_128/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_256/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_512/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_768/ --split_per=0.7 --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/OBSEA_1024/ --split_per=0.7 --eval-true --path_save=results/weights/