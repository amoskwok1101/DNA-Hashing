#!/bin/sh

export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

# cd /home/carloschau_prog2/amos/DAAE




trainFile=/home/amos/MgDB_amos/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_10000_testing
validFile=/home/amos/MgDB_amos/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_10000_testing

modelPath=/home/amos/MgDB_amos/checkpoints/dae_cosine_2

resume=0 # 0: start from scratch, 1: resume from last checkpoint or model
# if [ $resume -eq 0 ]; then
#     resume_wandb_id=$(/home/carloschau_prog2/miniconda3/envs/py38/bin/python init_wandb.py)
# fi
# resume=1
# resume_wandb_id=1fbbojc9
# resume_wandb_id=$(/home/carloschau_prog2/miniconda3/envs/py38/bin/python init_wandb.py)
mkdir -p $modelPath
while true; do
    echo "Start training..."
    python -m accelerate.commands.launch --num_processes=1  train.py --train $trainFile --valid $validFile \
        --model_type dae --epochs 10 --batch-size 512 \
        --dim_z 32 --save-dir $modelPath \
        --is-ladder --ladder-beta-type uniform --ladder-loss-type type_3 --lambda_adv 0 --lambda_sim 0 --lambda_margin 1 --lambda_kl 1 --lambda_quant 0.0 \
        --fixed-lambda-quant --rescaled-margin-type "quadratic" --similar-noise 0.03 --divergent-noise 0.2 \
        --loss-reduction mean --lr 0.0002 --distance_type cosine \
        --log-interval 50 \
        --model-path $modelPath --resume $resume \
        --no-Attention \
        --use-amp
        # --resume-wandb-id $resume_wandb_id 
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
