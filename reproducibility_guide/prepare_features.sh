#!/bin/bash

python3 generate_features.py --path=data/OBSEA_16/
python3 generate_features.py --path=data/OBSEA_32/
python3 generate_features.py --path=data/OBSEA_64/
python3 generate_features.py --path=data/OBSEA_128/
python3 generate_features.py --path=data/OBSEA_256/
python3 generate_features.py --path=data/OBSEA_512/
python3 generate_features.py --path=data/OBSEA_768/
python3 generate_features.py --path=data/OBSEA_1024/
