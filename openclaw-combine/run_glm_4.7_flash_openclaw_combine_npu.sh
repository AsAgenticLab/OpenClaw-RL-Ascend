#!/bin/bash
# =============================================================================
# OpenClaw-RL Personal Optimization — Combination Method (Binary RL + OPD)
# Ascend A3 single-node version (1 × 16 NPU)
#
# Usage (run on a single node):
#   1. Run this script directly (includes ray start --head internally)
#   2. ray job is automatically submitted at the end of this script
#
# Resource allocation (16 NPU total):
#   Actor (train)  : 8 NPU  (TP=4)
#   Rollout (infer): 4 NPU  (SGLang TP=1, 4 engines)
#   PRM (judge)    : 4 NPU  (SGLang TP=1, 4 engines)
# =============================================================================

# -----------------------------------------------------------------------------
# Clean up residual processes (reset environment for re-run)
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
# Point to slime-ascend (not slime)
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime-ascend" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"/workspace/Megatron-LM"}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-"/workspace/Megatron-Bridge/src"}
SGLANG_PYTHON_PATH=${SGLANG_PYTHON_PATH:-"/workspace/sglang/python"}
SLIME_ASCEND_ROOT=${SLIME_ASCEND_ROOT:-"/workspace/OpenClaw-RL/slime-ascend"}

source "${SLIME_DIR}/scripts/models/glm4.7-30B-A3B.sh"

# Modify the model paths below
HF_CKPT=${HF_CKPT:-/.../weights/GLM-4.7-Flash}
REF_LOAD=${REF_LOAD:-/.../weights/GLM-4.7-Flash_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-}

# =============================================================================
# OpenClaw API Server config
# These variables are read by openclaw_combine_api_server.py and must be
# included in RUNTIME_ENV_JSON
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
# Remote PRM config (optional)
# When set, judge/eval scoring is routed through a remote OpenAI-compatible API,
# leveraging a stronger model for evaluation. Teacher logprobs always use the
# local SGLang engine to ensure token-level alignment.
# =============================================================================
export OPENCLAW_REMOTE_PRM_BASE_URL="http://your-vllm-server:8000/v1"
export OPENCLAW_REMOTE_PRM_API_KEY="your-api-key"
export OPENCLAW_REMOTE_PRM_JUDGE_MODEL="qwen"   # Remote judge/eval model name
# =============================================================================
# Detailed PRM debug log (optional, disabled by default)
# When enabled, each PRM evaluation writes a complete record to a .jsonl file,
# including: timestamp, session_id, original question, original response,
# user feedback, raw PRM output, hint text, eval_score, etc.
# =============================================================================
export OPENCLAW_DETAILED_PRM_LOG=0
export OPENCLAW_DETAILED_PRM_LOG_FILE="${SCRIPT_DIR}/results/run_glm4.7_flash_detailed_prm.jsonl"  # Default path
# =============================================================================
# Create necessary directories
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
# Rollout parameters (OpenClaw async conversation-driven, no fixed dataset)
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
# Megatron parameters
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
# Combination Loss parameters (GRPO + OPD joint loss)
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
# Local PRM / Teacher engine parameters
# This SGLang engine computes teacher_log_probs for the OPD branch.
# The teacher model does NOT need to be identical to the student model;
# only the tokenizer / vocabulary must match for token-level alignment.
# Judge/eval scoring can be offloaded to a remote API via the
# OPENCLAW_REMOTE_PRM_* env vars (see remote PRM config section above).
# =============================================================================
PRM_ARGS=(
    --prm-enable
    --prm-num-gpus 4
    --prm-num-gpus-per-engine 4
    --prm-model-path "${HF_CKPT}"  # Model for teacher logprobs; only vocabulary must match the student
    --prm-m "${PRM_M:-1}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)


# =============================================================================
# Other Megatron parameters
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
# Custom function paths (OpenClaw Combine API service & reward function)
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
# Submit Ray Job
#
# PYTHONPATH notes:
#   - MEGATRON_LM_PATH       : Megatron-LM core (required)
#   - MEGATRON_BRIDGE_PATH   : Megatron-Bridge (required)
#   - SGLANG_PYTHON_PATH     : SGLang Python package (needed if not pip installed)
#   - SCRIPT_DIR             : openclaw-combine/ directory (contains combine_loss.py, etc.)
#   - openclaw-opd directory  : openclaw_opd_api_server.py (imported by combine)
#   - SLIME_ASCEND_ROOT      : slime-ascend package (needed if not pip install -e)
#
# RUNTIME_ENV_JSON env_vars are synced to all Ray worker nodes;
# all required NPU environment variables must be listed here.
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
    \"OPENCLAW_DETAILED_PRM_LOG\": \"${OPENCLAW_DETAILED_PRM_LOG:-}\",
    \"OPENCLAW_DETAILED_PRM_LOG_FILE\": \"${OPENCLAW_DETAILED_PRM_LOG_FILE:-}\"
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
