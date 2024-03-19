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
python train.py --train $trainFile --valid $validFile \
    --model_type aae --epochs 100 --batch-size 4096 \
    --dim_z 32 --save-dir $modelPath \
    --is-triplet --lambda_adv 0 --lambda_sim 0 --lambda_margin 1 \
    --similar-noise 0.05 --divergent-noise 0.5 \
    --log-interval 1000 --no-Attention \
    --lr 0.001 --distance_type cosine \
    #--load-model $modelPath/model.pt
    #--nlayers 2 \
    #--dim_emb 256 --dim_h 512 --dim_z 32 --dim_d 256 --nlayers 1 \


#vae
#python train.py --train $trainFile --valid $validFile \
#    --model_type vae --epochs 10 --batch-size 4096 \
#    --dim_z 32 --save-dir checkpoints/vae \
#    --is-triplet --lambda_kl 0.1 \
#    --similar-noise 0,0,0,0.1 --margin 0.2 \
#    --divergent-noise 0,0,0,0.2 \
#    --log-interval 1000 \
#    --lr 0.0001 \
    #--load-model checkpoints/dvae/model.pt
    #--is-binary
    
