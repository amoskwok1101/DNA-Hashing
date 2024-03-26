#!/bin/sh

export TOKENIZERS_PARALLELISM=true

#cd /home/d24h_prog2/yytao/SRA/DAAE
cd /container_data/ytye/SRA/codes/DAAE

#trainFile=./data/dna/10000000_64.txt
#validFile=./data/dna/100000_64.txt
#trainFile=./data/dna/4_64.txt
#validFile=./data/dna/4_64.txt

trainFile=../../data/virus_random_train.txt
#trainFile=../../real/virus_all.fasta_noAmbiguous_cdhit_onebase_90percent
validFile=../../data/virus_random_valid.txt
#validFile=../../real/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent

#reconstruction
#python train.py --train $trainFile --valid $validFile \
#    --model_type aae --lambda_adv 10 \
#    --dim_z 32 --noise 0,0,0,0.05 \
#    --save-dir checkpoints/daae \
#    --epochs 100 --batch-size 256
#    --load-model checkpoints/daae/model.pt

#aae
modelPath=checkpoints/aae32bits_cosine
if [ -f "$modelPath/model.pt" ]; then
    load_model_option="--load-model $modelPath/model.pt"
else
    load_model_option=""
fi
python -m accelerate.commands.launch  --num_processes=2  --mixed_precision=fp16 train_multi_gpu.py --train $trainFile --valid $validFile \
    --model_type aae --epochs 100 --batch-size 2048 \
    --dim_z 64 --save-dir $modelPath \
    --is-triplet --lambda_adv 0 --lambda_sim 0 --lambda_margin 1 \
    --similar-noise 0.05 \
    --divergent-noise 0.5 \
    --log-interval 2000 \
    --lr 0.00001 --distance_type cosine \
    --nlayers 1 \
    $load_model_option