#!/bin/bash

# Build base image
cd 0-base-images
docker build -t registry.gitlab.hpi.de/akita/i/python3-base:0.2.5 ./python3-base
docker build -t registry.gitlab.hpi.de/akita/i/python3-torch:0.2.5 ./python3-torch
docker build -t registry.gitlab.hpi.de/akita/i/pyod:0.2.5 ./pyod
docker build -t registry.gitlab.hpi.de/akita/i/python2-base:0.2.5 ./python2-base
docker build -t registry.gitlab.hpi.de/akita/i/python36-base:0.2.5 ./python36-base
cd ..

# Define the list of subfolders to ignore
ignored_folders=("01-data" "02-results" ".git" "0-base-images" "3-scripts")
supported_algs=("autoencoder" "cblof" "cof" "pcc" "hbos" "lof" "copod" "robust_pca" "random_black_forest" "dae")

# Get the list of sub-folders within the current folder, excluding the ones to ignore
subfolders=$(find . -mindepth 1 -maxdepth 1 -type d)

# Traverse through the list of sub-folders
for folder in $subfolders; do
	# Extract the name of the sub-folder
    folder_name=$(basename "$folder")
	# Check if the folder name is the list of supported algorithms
	if [[ "${supported_algs[*]}" =~ $folder_name ]]; then
		# Pass the sub-folder name to the desired command
		echo "$folder_name"
		docker build -t $folder_name ./$folder_name || true
	fi
	
done
