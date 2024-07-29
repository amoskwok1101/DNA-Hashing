#!/bin/bash

export TOKENIZERS_PARALLELISM=true
projectPath="/home/amos/MgDB_amos"
# modelPath="/home/amos/MgDB/checkpoints/aae_50mb_lstm_100_epoch"
# testFile="/home/amos/MgDB/data/6_merge_16.fasta_noAmbiguous_onebase_evenly_repeat100_testing"
modelName="dae_cosine"
modelPath="$projectPath/checkpoints/$modelName"
testFile="$projectPath/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_repeat100_testing"
# testFile="/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_repeat100_testing"
# testFile="/home/amos/MgDB/data/32bp_virus_species.fasta_noAmbiguous_rep_seq.fasta_cdhit80_onebase_10percent_evenly_repeat100_testing"
testPython="test.py"

un_embedding (){
    cd $projectPath
    noise_str=0,0,0,$1
    python $testPython --embedding --checkpoint $modelPath \
        --data $testFile --seed 20240320 --noise $noise_str \
         --median-file $medianFile
}

divergences=(0.05 0.1 0.15 0.2)
for divergence in "${divergences[@]}"; do
    run_embedding $divergence >dummy.txt
done
rm -f dummy.txt
cd $projectPath/result/ 

python tsne_plot.py --metrics "cosine,hamming" \
    --save-file "${modelName}-tsne.png"