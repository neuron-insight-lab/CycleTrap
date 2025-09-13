#!/bin/bash

model_name=("Qwen2.5-VL-7B" "InternVL3-8B" "LLaVa-Next" "LLaVa-Next-Video" "InstructBLIP" "InstructBlipVideo")
data_name=("coco" "imagenet" "tgif" "msvd")
GPU_ID=0

for model in "${model_name[@]}"; do
    for data in "${data_name[@]}"; do
        echo "Running attack with model: $model, dataset: $data"

        python main.py --model_name "$model" --data_name "$data" --sample_times 1  --device_id $GPU_ID
        python baseline/LingoLoop.py --model_name "$model" --data_name "$data" --sample_times 1  --device_id $GPU_ID
        python baseline/VerboseImage.py --model_name "$model" --data_name "$data" --sample_times 1  --device_id $GPU_ID
        python baseline/VerboseVideo.py --model_name "$model" --data_name "$data" --sample_times 1  --device_id $GPU_ID

    done
done    