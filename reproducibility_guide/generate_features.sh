#!/bin/bash

# An example command to generate feature using tsfresh
python3 generate_features.py --path=data/OBSEA_64/ --feature=tsfresh

# An example command to generate feature using catch22
python3 generate_features.py --path=data/OBSEA_128/ --feature=catch22
