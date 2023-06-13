#!/bin/bash

CONDA_PATH='/viscam/u/koven/mambdaforge/bin/activate'
ROOT_PATH='/viscam/projects/alp/code_release'
ENV_NAME='alp'

MESH_NAME="221106_pdiet"
MESH_PATH='/viscam/projects/alp/2022_ALP/nvdiffrec/out/'${MESH_NAME}'/mesh/mesh.obj'
INPUT_DIR='/viscam/projects/alp/2022_ALP/pre_exp/AlignHDRToolkit/testset'
SCENE_LIST=("indoor_04")
MODEL_NAME="pdiet"
OUTPUT_PATH="eval_results"

N_ROT=4
TRAIN_RES=2048

timestamp=$(date +%s)
SAVE_DIR=${OUTPUT_PATH}'/results_'${MESH_NAME}'_'${timestamp}

for scene in ${SCENE_LIST[@]}; do

    source ${CONDA_PATH}
    conda activate ${ENV_NAME}
    which python
    cd ${ROOT_PATH}

    echo 'processing '${INPUT_DIR}/${scene}/${MODEL_NAME}

    python run_sample.py --input_dir ${INPUT_DIR}/${scene}/${MODEL_NAME} --mesh_path ${MESH_PATH} --n_rot ${N_ROT} \
        --train_res ${TRAIN_RES} --output_dir ${SAVE_DIR}/${scene}/'best_pose'

    SCENE_PATH=${INPUT_DIR}/${scene}/${MODEL_NAME}
    OUTPUT_PATH=${SAVE_DIR}/${scene}/'best_pose'
    cp ${SCENE_PATH}'/hdr_rgba.exr' ${OUTPUT_PATH}'/hdr.exr'

    python optimize_envmap.py \
    --input_dir $OUTPUT_PATH --batch_size 2000000 --n_samples 1 --repr_emap_in_cam --patience 200 --n_iters 10000

    python shift_exr.py --input_dir $OUTPUT_PATH --input_filename emap_learned.exr

done


