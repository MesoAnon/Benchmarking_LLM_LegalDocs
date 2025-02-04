# !/bin/sh
trap "kill 0" EXIT

dataset=$1
model=$2
port=$3
cuda_id=$4

export CUDA_VISIBLE_DEVICES=$cuda_id

declare -A model_map=( ["qwen"]="Qwen/Qwen2-7B-Instruct" ["phi3"]="microsoft/Phi-3-small-128k-instruct" )
declare -A model_path_map=( ["qwen"]="Qwen2-7B-Instruct" ["phi3"]="Phi-3-small-128k-instruct" )
declare -A summary_length=( ["multilex_tiny"]=25 ["multilex_short"]=130 ["multilex_long"]=650 ["eurlexsum"]=600 ["eurlexsum_test"]=600 ["eurlexsum_validation"]=600 )

# if you download the model to a custom location, use version 1), else use version 2)
# 1)
python -m vllm.entrypoints.openai.api_server --served-model-name ${model_map[$model]} --dtype auto --api-key NONE --trust-remote-code --max-model-len 131072 --model models/${model_path_map[$model]} --port $port & 
# python -m vllm.entrypoints.openai.api_server --served-model-name ${model_map[$model]} --dtype bfloat16 --api-key NONE --trust-remote-code --max-model-len 131072 --model models/${model_path_map[$model]} --port $port --gpu-memory-utilization 0.99 --enforce-eager & 

# 
# 

# 2)
# python -m vllm.entrypoints.openai.api_server --model ${model_map[$model]} --dtype auto --api-key NONE --trust-remote-code --max-model-len 131072
echo "Server is starting, please wait..."
while ! curl localhost:$port -m1 -o /dev/null -s; do
    sleep 30
    echo "Server is still starting..."
done
echo "Server is running!"

python vllm_summary.py --model $model --dataset $dataset --word_count ${summary_length[$dataset]} --port $port

 