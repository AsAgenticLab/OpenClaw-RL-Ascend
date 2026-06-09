#!/bin/bash
# =============================================================================
# OpenClaw-RL Personal Optimization — Combination Method (Binary RL + OPD)
# 昇腾 A3 双机适配版本 (2 × 16 NPU = 32 NPU)
#
# 用法（在 Node0 主节点上执行）：
#   1. 先在 Node0 上运行本脚本（内含 ray start --head）
#   2. 在 Node1 从节点上运行：
#        ray start --address=<NODE0_IP>:6379 --num-gpus 16
#   3. ray job 已在本脚本末尾自动提交
#
# 资源分配（32 NPU 总计）：
#   Actor (训练)  : 2 nodes × 8 NPU = 16 NPU  (TP=4, 2 DP/node)
#   Rollout (推理): 8 NPU                       (SGLang TP=2, 4 engines)
#   PRM (评判)    : 8 NPU                       (SGLang TP=2, 4 engines)
# =============================================================================

# -----------------------------------------------------------------------------
# 清理残留进程（重新运行时复位环境）
# 注意：Node1 需在本脚本 ray start --head 之后、ray job submit 之前手动加入
# -----------------------------------------------------------------------------
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
unset http_proxy https_proxy
set -ex

export PYTHONBUFFERED=16
export PYTHONPATH="/workspace/Megatron-LM/:/workspace/sglang/python:$PYTHONPATH"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEBUG=1
export RAY_DEDUP_LOGS=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Increase Ray heartbeat/health-check timeouts to reduce false node failures under heavy init.
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# 指向 slime-ascend（而非 slime）
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime-ascend" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"/workspace/Megatron-LM"}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-"/workspace/Megatron-Bridge/src"}
SGLANG_PYTHON_PATH=${SGLANG_PYTHON_PATH:-"/workspace/sglang/python"}
SLIME_ASCEND_ROOT=${SLIME_ASCEND_ROOT:-"/workspace/OpenClaw-RL/slime-ascend"}

source "${SLIME_DIR}/scripts/models/glm4.7-30B-A3B.sh"

# 修改如下的模型位置
HF_CKPT=${HF_CKPT:-/.../weights/GLM-4.7-Flash}
REF_LOAD=${REF_LOAD:-/.../weights/GLM-4.7-Flash_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-}

# =============================================================================
# OpenClaw API Server 配置
# 这些变量由 openclaw_combine_api_server.py 读取，必须包含在 RUNTIME_ENV_JSON 中
# =============================================================================
export SGLANG_API_KEY="${SGLANG_API_KEY:-sk-1234}"
export SERVED_MODEL_NAME="glm-4.7-flash"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/glm4.7_flash_record.jsonl"
export TP="1"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.8"
export REASONING_PARSER="qwen3"
# GLM-4.7 generates XML-style tool calls (<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>)
# which is not supported by any built-in SGLang parser.  Set qwen25 so SGLang at least attempts
# structured parsing; the openclaw proxy layer has a GLM-XML fallback that converts the XML format
# into proper OpenAI tool_calls when qwen25 finds nothing.
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"
# =============================================================================
# 远程 PRM 配置（可选，设置后将使用远程 OpenAI-compatible API 替代本地 PRM）
# =============================================================================
# export OPENCLAW_REMOTE_PRM_BASE_URL="http://your-vllm-server:8000/v1"
# export OPENCLAW_REMOTE_PRM_API_KEY="your-api-key"
# export OPENCLAW_REMOTE_PRM_JUDGE_MODEL="qwen2.5-72b-instruct"   # 用于 judge + eval 的模型
# export OPENCLAW_REMOTE_PRM_TEACHER_MODEL=""                       # 用于 teacher logprobs（需与 student 同 tokenizer；不设则用本地）
# =============================================================================
# 创建必要目录
# =============================================================================
mkdir -p "${SCRIPT_DIR}/results"
#mkdir -p "${SAVE_CKPT}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${REF_LOAD}"
    #--save "${SAVE_CKPT}"
    #--save-interval 100
    #--rotary-base 5000000
)

# =============================================================================
# Rollout 参数（OpenClaw 异步对话驱动，无固定数据集）
# =============================================================================
ROLLOUT_ARGS=(
    --disable-rollout-global-dataset
    --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine

    --num-rollout 100000000
    --rollout-batch-size 16
    --n-samples-per-prompt 1
    --rollout-max-response-len 4096
    --rollout-max-context-len 8192
    --rollout-temperature 1
    --reward-key score

    --num-steps-per-rollout 1
)

# =============================================================================
# Megatron 参数
# TP4 DP1
# =============================================================================
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-chunk-size 1024
)

# =============================================================================
# Combination Loss 参数（GRPO + OPD 联合损失）
# =============================================================================
COMBINE_ARGS=(
    --advantage-estimator grpo
    --disable-rewards-normalization
    --loss-type custom_loss
    --custom-loss-function-path combine_loss.combine_loss_function
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 7e-7
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.88
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-cuda-graph-max-bs 16
   --sglang-max-running-requests 64
   --sglang-device npu
)

# =============================================================================
# PRM 参数，需修改模型位置
# =============================================================================
PRM_ARGS=(
    --prm-enable
    --prm-num-gpus 4
    --prm-num-gpus-per-engine 4
    --prm-model-path /.../weights/GLM-4.7-Flash
    --prm-m "${PRM_M:-1}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)


# =============================================================================
# 其他 Megatron 参数
# =============================================================================
MISC_ARGS=(
    --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --use-flash-attn
)

# =============================================================================
# 自定义函数路径（OpenClaw Combine API 服务 & 奖励函数）
# =============================================================================
CUSTOM_ARGS=(
    --custom-generate-function-path openclaw_combine_api_server.generate
    --custom-rm-path openclaw_combine_api_server.reward_func
)


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR},localhost"
ray start --head --node-ip-address ${MASTER_ADDR} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265



# =============================================================================
# 提交 Ray Job
#
# PYTHONPATH 说明：
#   - MEGATRON_LM_PATH       : Megatron-LM 核心（必须）
#   - MEGATRON_BRIDGE_PATH   : Megatron-Bridge（必须）
#   - SGLANG_PYTHON_PATH     : SGLang Python 包（若未 pip install 则需要）
#   - SCRIPT_DIR             : openclaw-combine/ 目录（含 combine_loss.py 等）
#   - openclaw-opd 目录       : openclaw_opd_api_server.py（被 combine 导入）
#   - SLIME_ASCEND_ROOT      : slime-ascend 包（若未 pip install -e 则需要）
#
# 多机注意：RUNTIME_ENV_JSON 中的 env_vars 会同步到所有 Ray worker 节点，
#           包括 Node1，因此所有必需的 NPU 环境变量都必须列在这里。
# =============================================================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${MEGATRON_BRIDGE_PATH}:${SGLANG_PYTHON_PATH}:${SCRIPT_DIR}:${SCRIPT_DIR}/../openclaw-opd:${SLIME_ASCEND_ROOT}:${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_NPU_ALLOC_CONF\": \"expandable_segments:True\",
    \"ASCEND_RT_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"HCCL_HOST_SOCKET_PORT_RANGE\": \"60000-60050\",
    \"HCCL_NPU_SOCKET_PORT_RANGE\": \"61000-61050\",
    \"ASCEND_TOOLKIT_HOME\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_OPP_PATH\": \"/usr/local/Ascend/cann-8.5.0/opp/\",
    \"ASCEND_AICPU_PATH\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_HOME_PATH\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"SGLANG_API_KEY\": \"${SGLANG_API_KEY}\",
    \"SERVED_MODEL_NAME\": \"${SERVED_MODEL_NAME}\",
    \"HOST\": \"${HOST}\",
    \"PORT\": \"${PORT}\",
    \"TP\": \"${TP}\",
    \"CONTEXT_LENGTH\": \"${CONTEXT_LENGTH}\",
    \"MEM_FRACTION_STATIC\": \"${MEM_FRACTION_STATIC}\",
    \"REASONING_PARSER\": \"${REASONING_PARSER}\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER}\",
    \"PRM_M\": \"${PRM_M}\",
    \"OPENCLAW_RECORD_ENABLED\": \"${OPENCLAW_RECORD_ENABLED}\",
    \"OPENCLAW_RECORD_FILE\": \"${OPENCLAW_RECORD_FILE}\",
    \"OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY\": \"${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY}\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\",
    \"OPENCLAW_REMOTE_PRM_BASE_URL\": \"${OPENCLAW_REMOTE_PRM_BASE_URL:-}\",
    \"OPENCLAW_REMOTE_PRM_API_KEY\": \"${OPENCLAW_REMOTE_PRM_API_KEY:-}\",
    \"OPENCLAW_REMOTE_PRM_JUDGE_MODEL\": \"${OPENCLAW_REMOTE_PRM_JUDGE_MODEL:-}\",
    \"OPENCLAW_REMOTE_PRM_TEACHER_MODEL\": \"${OPENCLAW_REMOTE_PRM_TEACHER_MODEL:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ASCEND_ROOT}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 4 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${COMBINE_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]} \
    ${PRM_ARGS[@]} | tee logs/openclaw-combine-glm-4.7.log
