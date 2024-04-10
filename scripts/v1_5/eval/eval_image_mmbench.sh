:#!/bin/bash

SPLIT="mmbench_dev_20230712"

CKPT_NAME="Video-LLaVA-7B"
CKPT="LanguageBind/${CKPT_NAME}"
EVAL="eval"

echo Starting Step 1
python3 -m videollava.eval.model_vqa_mmbench \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
echo Creating Dir
mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT
print("Doing thing for submissions")
python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}
echo Sucessfully completed
