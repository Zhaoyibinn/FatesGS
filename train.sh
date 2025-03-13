SCAN="$1"
EXP_NAME="$2"
SOURCE_PATH="data/DTU/set_23_24_33/scan${SCAN}"
MODEL_PATH="output/DTU/set_23_24_33/${EXP_NAME}/scan${SCAN}"

CUDA_VISIBLE_DEVICES=$3 python train.py -s $SOURCE_PATH -m $MODEL_PATH -r 2
CUDA_VISIBLE_DEVICES=$3 python render.py -s $SOURCE_PATH -m $MODEL_PATH -r 2 --skip_test --skip_train
