#!/bin/bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Stage 1: Build disease diagnosis datasets"
echo "Datasets: CMD, DXY, GMD"
echo "=========================================="

python "${SCRIPT_DIR}/build_disease_dataset.py" \
    --datasets CMD DXY GMD \
    --do_preprocess_dataset true \
    --do_preprocess_turn true \
    --use_branch false \
    --use_repeat false \
    --include_initial_state true \
    --max_aug_depth 10

echo ""
echo "=========================================="
echo "Stage 2: Build symptom completion datasets"
echo "=========================================="

# --- CMD: single_order, no repeat augmentation ---
echo ""
echo "--- CMD ---"
python "${SCRIPT_DIR}/build_symptom_dataset.py" \
    --gmd_data_dir "${SCRIPT_DIR}/raw_data/CMD" \
    --out_dir "${SCRIPT_DIR}/processed_data/diffusion_model_data/CMD" \
    --mode single_order \
    --position_repeat 0 \
    --position_repeat_base 1 \
    --target_shuffle_num 0 \
    --position_repeat_direction ascending \
    --rare_first_shuffle_num 0 \
    --include_false_explicit 1 \
    --include_false_implicit 1 \
    --seed 42

# --- DXY: repeated_permutation, position repeat augmentation ---
echo ""
echo "--- DXY ---"
python "${SCRIPT_DIR}/build_symptom_dataset.py" \
    --gmd_data_dir "${SCRIPT_DIR}/raw_data/DXY" \
    --out_dir "${SCRIPT_DIR}/processed_data/diffusion_model_data/DXY" \
    --mode repeated_permutation \
    --repeat_num 4 \
    --position_repeat 1 \
    --position_repeat_base 0 \
    --target_shuffle_num 1 \
    --position_repeat_direction ascending \
    --rare_first_shuffle_num 0 \
    --include_false_explicit 1 \
    --include_false_implicit 1 \
    --seed 42

# --- GMD: repeated_permutation, position repeat augmentation ---
echo ""
echo "--- GMD ---"
python "${SCRIPT_DIR}/build_symptom_dataset.py" \
    --gmd_data_dir "${SCRIPT_DIR}/raw_data/GMD" \
    --out_dir "${SCRIPT_DIR}/processed_data/diffusion_model_data/GMD" \
    --mode repeated_permutation \
    --repeat_num 4 \
    --position_repeat 1 \
    --position_repeat_base 0 \
    --target_shuffle_num 1 \
    --position_repeat_direction ascending \
    --rare_first_shuffle_num 0 \
    --include_false_explicit 1 \
    --include_false_implicit 1 \
    --seed 42

echo ""
echo "=========================================="
echo "All datasets built successfully."
echo "=========================================="
