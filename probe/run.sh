#!/usr/bin/env bash
#
# run.sh - Extract features, train probe, and test for CMD / DXY / GMD sequentially
#
# Usage:
#   cd probe
#   bash run.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_ROOT="${PROJECT_ROOT}/data/processed_data/disease_data"
TURN_DATA_DIR="${DATA_ROOT}/turn_data"
FEATURE_ROOT="${PROJECT_ROOT}/re/probe_features"
PROBE_ROOT="${PROJECT_ROOT}/re/probe_models"

# Model path (user must set this)
# LLM model path (required, e.g.: export MODEL_PATH=/path/to/model)
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: Please set MODEL_PATH env var to LLM model directory"
    echo "Usage: MODEL_PATH=/path/to/model bash run.sh"
    exit 1
fi

for DATASET in CMD DXY GMD; do
    echo ""
    echo "############################################################"
    echo "# Probe: ${DATASET}"
    echo "############################################################"

    FEATURE_DIR="${FEATURE_ROOT}/${DATASET}"
    PROBE_DIR="${PROBE_ROOT}/${DATASET}"
    TURN_FILE="${TURN_DATA_DIR}/${DATASET}_train_turn.json"
    TEST_FILE="${DATA_ROOT}/${DATASET}/test.json"
    DISEASE_CORPUS="${DATA_ROOT}/${DATASET}/disease_corpurs.txt"

    CONFIG="${PROBE_DIR}/config.json"
    mkdir -p "${PROBE_DIR}"

    cat > "${CONFIG}" <<EOF
{
  "mode": "all",
  "seed": 42,
  "paths": {
    "model_name_or_path": "${MODEL_PATH}",
    "data_root": "${DATA_ROOT}",
    "dataset_name": "${DATASET}",
    "turn_data_file": "${TURN_FILE}",
    "test_data_file": "${TEST_FILE}",
    "disease_corpus_file": "${DISEASE_CORPUS}",
    "feature_dir": "${FEATURE_DIR}",
    "probe_dir": "${PROBE_DIR}"
  },
  "split": {
    "val_ratio": 0.1,
    "test_ratio": 0.1
  },
  "prompt": {
    "symptom_source": "current",
    "disease_set_source": "all",
    "system_prompt": "",
    "user_template": "## 患者临床表现\\n- 阳性症状：{positive_symptoms}。\\n- 阴性症状：{negative_symptoms}。\\n\\n## 候选疾病\\n{疾病集合}\\n\\n请结合患者的阳性与阴性症状，在候选疾病中选择最可能的诊断。请仅输出疾病名称，不要输出其他内容。"
  },
  "extract": {
    "device": "cuda",
    "torch_dtype": "bf16",
    "use_4bit": false,
    "max_length": 2048,
    "batch_size": 16,
    "pooling": "mean",
    "layer_indices": [-1, -2, -3, -4],
    "normalize": true,
    "add_generation_prompt": true,
    "padding_side": "left",
    "splits": ["train", "val", "test"],
    "resume": true
  },
  "pca": {
    "enabled": true,
    "n_components": 256,
    "whiten": false
  },
  "probe": {
    "type": "linear",
    "hidden_dim": 512,
    "dropout": 0.1,
    "lr": 0.01,
    "weight_decay": 0.01,
    "batch_size": 64,
    "epochs": 500,
    "patience": 500,
    "class_weight": "auto",
    "select_metric": "macro_f1",
    "topk": [1, 3, 5],
    "temperature": 1.0,
    "max_grad_norm": 5.0
  },
  "test": {
    "symptoms_key": "all_symptoms",
    "max_test_samples": -1
  }
}
EOF

    echo "[Config] ${CONFIG}"
    echo "  turn_data_file: ${TURN_FILE}"
    echo "  test_data_file:  ${TEST_FILE}"
    echo "  feature_dir:     ${FEATURE_DIR}"
    echo "  probe_dir:       ${PROBE_DIR}"

    python "${SCRIPT_DIR}/disease_probe_train.py" \
        --config "${CONFIG}" \
        --mode all
done

echo ""
echo "All probes done. Results saved to ${PROBE_ROOT}"
