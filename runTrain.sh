#!/bin/sh

export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

# cd /home/carloschau_prog2/amos/DAAE



# trainFile=/home/carloschau_prog2/amos/DAAE/data/virus_all.fasta_noAmbiguous_cdhit_onebase_90percent
# trainFile=/home/carloschau_prog2/amos/DAAE/data/6_merge_16.fasta_noAmbiguous_onebase_90percent
# trainFile=/home/carloschau_prog2/amos/DAAE/data/merge_112.fasta_rep_seq.fasta_onebase_90percent
# trainFile=/home/carloschau_prog2/tracy/data/combined_2.8.fasta_noAmbiguous_cdhit_onebase_90percent
# validFile=/home/carloschau_prog2/tracy/data/combined_2.8.fasta_noAmbiguous_cdhit_onebase_10percent
# trainFile=/home/carloschau_prog2/tracy/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_90percent
# validFile=/home/carloschau_prog2/tracy/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent
# trainFile=/home/carloschau_prog2/amos/DAAE/data/100bp_species_virus_seq.fasta_90percent
# validFile=/home/carloschau_prog2/amos/DAAE/data/100bp_species_virus_seq.fasta_10percent
# validFile=/home/carloschau_prog2/amos/DAAE/data/merge_112_10percent.fasta_onlyseq
# validFile=/home/carloschau_prog2/amos/DAAE/data/virus_all.fasta_noAmbiguous_10percent
# validFile=/home/carloschau_prog2/amos/DAAE/data/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent
# validFile=/home/carloschau_prog2/amos/DAAE/data/merge_112.fasta_rep_seq.fasta_onebase_10percent
# validFile=/home/carloschau_prog2/amos/DAAE/data/6_merge_16.fasta_noAmbiguous_onebase_10percent
trainFile=/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_10000_testing
validFile=/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_1000_testing
# trainFile=/home/amos/MgDB/data/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent_train
# validFile=/home/amos/MgDB/data/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent_test
# modelPath=/home/carloschau_prog2/amos/DAAE/checkpoints/aae_lstm_cnn_cosine_2_8GB_sigmoid_masked_beta_inc
modelPath=/home/amos/MgDB/checkpoints/dae_cosine_ladder_pearson

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
        --model_type dae --epochs 10 --batch-size 512\
        --dim_z 32 --save-dir $modelPath \
        --is-ladder --ladder-pearson --lambda_adv 0 --lambda_sim 0 --lambda_margin 1 --lambda_kl 1 --lambda_quant 0.0000001 \
        --fixed-lambda-quant --rescaled-margin-type "scaled to dim_z" --similar-noise 0.03 --divergent-noise 0.2 \
        --lr 0.0001 --distance_type cosine \
        --log-interval 100 \
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
