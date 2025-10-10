#!/bin/bash

# DATA_FILE="$1" # 待评估模型结果文件
MAX_TEST_IMG_NUM=${1:-0} # 最多评估的图片数量,0代表全部测试
MODEL_NAME=${2:-'qwen2.5-vl-72b-instruct'} # 模型名称
API_URL=${3:-'https://api.qingyuntop.top/v1'} # API的URL
API_KEY=${4:-'sk-mFtES1U9ZQqCpLuoODW1cH6XyDOZEMcfKzBNSq9ROBEBV5YW'} # API的令牌
SCORE_TIMES=${5:-1} # 对一次评估结果的评分次数

# 从DATA_FILE路径中提取最后两级目录作为输出目录名
get_output_dir() {
    local input_file="$1"
    local dir_path=$(dirname "$input_file")
    local parent_dir=$(basename "$dir_path")
    local filename=$(basename "$input_file" .xlsx)
    # 获取时间信息
    timestamp_short=$(date +%Y%m%d%H%M%S)
    echo "eval_data/llm_score/api_results/${timestamp_short}_parallel_results_${parent_dir}_${filename}"
}

# 获取脚本的绝对路径
SCRIPT_ABS_PATH="$(readlink -f "$0")"

# 获取脚本所在的目录
SCRIPT_DIR="$(dirname "$SCRIPT_ABS_PATH")"

echo "脚本所在的绝对路径: $SCRIPT_DIR"
# 获取evaluation目录
evaluation_dir=$(dirname "$(dirname "$SCRIPT_DIR")")
echo "evaluation目录: $evaluation_dir"
# 获取api测试脚本路径
API_TEST_SCRIPT_PATH="$evaluation_dir/code/02_llm_ability_eval/get_foodname_parallel_by_api.py"
echo "API测试脚本路径: $API_TEST_SCRIPT_PATH"

# 执行模型评估
python $API_TEST_SCRIPT_PATH \
  --api_url "$API_URL" \
  --api_key "$API_KEY" \
  --model_name "$MODEL_NAME" \
  --max_test_img_num $MAX_TEST_IMG_NUM \
  --output_path "$evaluation_dir/eval_data/eval_res_of_llm/"

# 取输出路径下最新的文件作为数据文件
DATA_FILE=$evaluation_dir/eval_data/eval_res_of_llm/$(ls -t $evaluation_dir/eval_data/eval_res_of_llm/ | head -n 1)
echo "评估结果文件: $DATA_FILE"

# 对一次评估结果反复打分
for ((i=0; i<$SCORE_TIMES; i++)); do
  echo "第 $i 次评分..."

  # 输出目录
  OUT_DIR=$evaluation_dir/$(get_output_dir "$DATA_FILE")
  echo "输出目录: $OUT_DIR"

  NUM_SAMPLES=50 # 50条一组

  mkdir -p "$OUT_DIR"

  TOTAL=500 # 评估数据条数

  NUM_BATCHES=$(( ($TOTAL + $NUM_SAMPLES - 1) / $NUM_SAMPLES ))

  for i in $(seq 0 $((NUM_BATCHES-1))); do
    START=$((i * NUM_SAMPLES))
    OUT_FILE="${OUT_DIR}/eval_${START}.xlsx"
    echo "启动第$i批：样本 $START ~ $((START+NUM_SAMPLES-1))"
    nohup python $SCRIPT_DIR/main.py --start $START --num $NUM_SAMPLES --input "$DATA_FILE" --output "$OUT_FILE" >> "${OUT_DIR}/log_${i}.txt" 2>&1 &
  done

  wait
  echo "第 $i 次评分所有批处理完成"

  python $SCRIPT_DIR/merge.py --input-dir "$OUT_DIR"
done