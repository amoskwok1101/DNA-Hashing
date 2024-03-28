#!/bin/sh

export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

cd /media/data/amos/DAAE

trainFile=/media/data/amos/DAAE/data/seq_tiny.txt
validFile=/media/data/amos/DAAE/data/seq_tiny.txt

modelPath=/media/data/amos/DAAE/checkpoints/aae_test_multi_gpu
resume=0 # 0: start from scratch, 1: resume from last checkpoint or model
if [ $resume -eq 0 ]; then
    resume_wandb_id=$(python init_wandb.py)
fi
# resume=1
# resume_wandb_id=1fbbojc9
# resume_wandb_id=$(python init_wandb.py)

while true; do
    echo "Start training..."
    python -m accelerate.commands.launch --num_processes=1  --mixed_precision=fp16 train_multi_gpu.py --train $trainFile --valid $validFile \
        --model_type aae --epochs 3 --batch-size 1024 \
        --dim_z 32 --save-dir $modelPath \
        --is-triplet --lambda_adv 0 --lambda_sim 0 --lambda_margin 2 \
        --similar-noise 0.05 --divergent-noise 0.2 \
        --lr 0.001 --distance_type cosine \
        --log-interval 1000 \
        --model-path $modelPath --resume $resume \
        --use_transformer
        # --use-wandb --resume-wandb-id $resume_wandb_id 
        #--load-model checkpoints/paae/model.pt
        #--dim_emb 256 --dim_h 512 --dim_z 32 --dim_d 256 --nlayers 1 \
            
    exit_status=$?

    # Assuming exit status 0 means success and anything else means failure
    if [ $exit_status -eq 0 ]; then
        echo "Training completed successfully."
        break
    else
        echo "Restarting..."
        resume=1
        # Optionally, include a sleep command to pause before restarting
        sleep 60
    fi
done
