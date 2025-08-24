# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"

## Qwen2.5-7B-instruct
export CUDA_VISIBLE_DEVICES="1,2,3,4"
MODEL_NAME_OR_PATH="../../Models/qwen/Qwen2.5-7B_lora_pretrain_merge_Alphaca"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

## Qwen2.5-3B-instruct
#MODEL_NAME_OR_PATH="../../Models/qwen/Qwen2.5-7B_lora_pretrain_merge"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

## Qwen2.5-Math-1.5B-Instruct
#export CUDA_VISIBLE_DEVICES="0"
#MODEL_NAME_OR_PATH="../../distill-code/results/qwen2.5-instruct/train/sft/e10-bs1-lr1e-05-G2-N2-NN1/"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

## Qwen2.5-Math-72B-Instruct
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH



## Evaluate Qwen2-Math-Instruct
#PROMPT_TYPE="qwen-boxed"
#
## Qwen2-Math-1.5B-Instruct
#export CUDA_VISIBLE_DEVICES="0"
#MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
#
## Qwen2-Math-7B-Instruct
#export CUDA_VISIBLE_DEVICES="0"
#MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
#
## Qwen2-Math-72B-Instruct
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
#bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
