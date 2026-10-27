#!/usr/bin/env bash
#
# run.sh - Train and evaluate diffusion model for CMD / DXY / GMD sequentially
#
# Usage:
#   cd Diffusion
#   bash run.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_ROOT="${PROJECT_ROOT}/data/processed_data/diffusion_model_data"
SAVE_ROOT="${PROJECT_ROOT}/saved_models"

STEPS_LIST=(40)

for DATASET in CMD DXY GMD; do
    DATA_PATH="${DATA_ROOT}/${DATASET}"
    echo ""
    echo "############################################################"
    echo "# Dataset: ${DATASET}"
    echo "# Data:    ${DATA_PATH}"
    echo "############################################################"

    for STEPS in "${STEPS_LIST[@]}"; do
        echo ""
        echo "=============================================="
        echo "Training ${DATASET} with steps=${STEPS}"
        echo "=============================================="

        python "${SCRIPT_DIR}/main.py" \
            --dataset "${DATASET}" \
            --data_path "${DATA_PATH}" \
            --emb_path "${DATA_ROOT}" \
            --cuda \
            --gpu '0' \
            --epochs 150 \
            --batch_size 64 \
            --n_cate 1 \
            --in_dims '[64]' \
            --out_dims '[]' \
            --mlp_dims '[64]' \
            --emb_size 10 \
            --mean_type x0 \
            --steps "${STEPS}" \
            --noise_scale 0.1 \
            --noise_min 0.0001 \
            --noise_max 0.02 \
            --sampling_noise False \
            --valid_ratio 0.1 \
            --topN '[1,3,5,10]' \
            --log_name "log_step${STEPS}" \
            --save_path "${SAVE_ROOT}"

        echo ""
        echo "--- Inference ${DATASET} steps=${STEPS} ---"

        python "${SCRIPT_DIR}/inference.py" \
            --dataset "${DATASET}" \
            --data_path "${DATA_PATH}" \
            --cuda \
            --gpu '0' \
            --steps "${STEPS}" \
            --mean_type x0 \
            --noise_scale 0.1 \
            --noise_min 0.0001 \
            --noise_max 0.02 \
            --sampling_noise False \
            --eval_test \
            --topN '[1,3,5,10]' \
            --model_path "${SAVE_ROOT}/${DATASET}/${DATASET}_log_step${STEPS}_best.pth" \
            --ae_path "${SAVE_ROOT}/${DATASET}/${DATASET}_log_step${STEPS}_best_AE.pth" \
            --cond_path "${SAVE_ROOT}/${DATASET}/${DATASET}_log_step${STEPS}_best_COND.pth"
    done
done

echo ""
echo "All datasets done. Models saved to ${SAVE_ROOT}"
