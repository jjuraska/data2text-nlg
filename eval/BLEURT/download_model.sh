#!/bin/sh

#MODEL_NAME="BLEURT-20"
#MODEL_NAME="BLEURT-20-D12"
#MODEL_NAME="BLEURT-20-D6"
MODEL_NAME="BLEURT-20-D3"

wget https://storage.googleapis.com/bleurt-oss-21/$MODEL_NAME.zip -P ./models/
unzip ./models/$MODEL_NAME.zip -d ./models/
rm ./models/$MODEL_NAME.zip
