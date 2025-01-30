#!/usr/bin/env bash

# download taobao dataset from Kaggle
echo Downloading Taobao Dataset

OUTPUT_DIR=dataset/taobao/
DOWNLOAD_FILE=taobao.zip
mkdir -p $OUTPUT_DIR
curl -L -o $DOWNLOAD_FILE https://www.kaggle.com/api/v1/datasets/download/pavansanagapati/ad-displayclick-data-on-taobaocom
unzip -o $DOWNLOAD_FILE -d $OUTPUT_DIR
rm $DOWNLOAD_FILE

echo Finished download of Taobao Dataset

echo Downloading MovieLens Dataset

OUTPUT_DIR=dataset/movielens-20/
DOWNLOAD_FILE=movielens.zip
mkdir -p $OUTPUT_DIR
curl -L -o $DOWNLOAD_FILE https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip -j -o $DOWNLOAD_FILE -d $OUTPUT_DIR
rm $DOWNLOAD_FILE

echo Finished download of MovieLens Dataset