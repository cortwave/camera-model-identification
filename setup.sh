#!/usr/bin/env bash

git clone https://github.com/Cadene/pretrained-models.pytorch.git

mkdir data
mkdir results

cd data

kg download -u '$1' -p '$2' -c 'sp-society-camera-model-identification'

unzip test.zip
unzip train.zip

cd ././src

python3 scripts/split_folds.py

echo 'Done!'
