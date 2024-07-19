#!/bin/bash

export TOKENIZERS_PARALLELISM=true
projectPath="/home/amos/MgDB"
# modelPath="/home/amos/MgDB/checkpoints/aae_50mb_lstm_100_epoch"
# testFile="/home/amos/MgDB/data/6_merge_16.fasta_noAmbiguous_onebase_evenly_repeat100_testing"
modelName="dae_cosine_ladder_pearson"
modelPath="/home/amos/MgDB/checkpoints/$modelName"
# validFile="$projectPath/data/32bp_virus_species.fasta_noAmbiguous_rep_seq.fasta_cdhit80_onebase_10percent_evenly_100000_testing"
# testFile="$projectPath/data/32bp_virus_species.fasta_noAmbiguous_rep_seq.fasta_cdhit80_onebase_10percent_evenly_10000_testing"
# validFile="$projectPath/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_100000_testing"
# testFile="$projectPath/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_10000_testing"
validFile="/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_10000_testing"
testFile="/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_10000_testing"
# validFile="/home/carloschau_prog2/amos/DAAE/data/5000seq.txt"
# testFile="/home/carloschau_prog2/amos/DAAE/data/5000seq.txt"
# validFile="/home/carloschau_prog2/amos/DAAE/data/merge_112.fasta_rep_seq.fasta_onebase_90percent_evenly_10000_testing"
# testFile="/home/carloschau_prog2/amos/DAAE/data/merge_112.fasta_rep_seq.fasta_onebase_90percent_evenly_10000_testing"
# validFile=/home/amos/MgDB/data/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent_test
testPython="test.py"

medianFile="$modelName-median.pkl"

run_embedding_all (){
    cd $projectPath
    python $testPython --embedding-all --checkpoint $modelPath \
        --data $validFile --seed 20240320 --median-file $medianFile
}

run_embedding_all

echo "Run test retrieval"
run_test_retrieval (){
    cd $projectPath
    python $testPython --test-retrieval --checkpoint $modelPath \
    --data $testFile --seed 20240320 --use-median --median-file $medianFile
}

run_test_retrieval 
# > "${projectPath}/retrieval.csv"
run_test_retrieval_by_radius (){
    cd $projectPath
    python $testPython --retrieval-by-radius --checkpoint $modelPath \
    --data $validFile --seed 20240320 --use-median --median-file $medianFile
}
run_test_retrieval_by_radius

# cd /home/carloschau_prog2/amos/DAAE/result
# /home/carloschau_prog2/miniconda3/envs/py38/bin/python hit_minus_top1_distribution_plot.py
# /home/carloschau_prog2/miniconda3/envs/py38/bin/python retrieval_by_radius.py