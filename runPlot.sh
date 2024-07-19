#!/bin/bash

export TOKENIZERS_PARALLELISM=true
projectPath="/home/amos/MgDB/"
# modelPath="/home/amos/MgDB/checkpoints/aae_50mb_lstm_100_epoch"
# testFile="/home/amos/MgDB/data/6_merge_16.fasta_noAmbiguous_onebase_evenly_repeat100_testing"
modelName="dae_cosine_ladder_quant"
modelPath="/home/amos/MgDB/checkpoints/$modelName"
testFile="/home/amos/MgDB/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_repeat100_testing"
# testFile="/home/amos/MgDB/data/100bp_species_virus_seq.fasta_10percent_evenly_repeat100_testing"
# testFile="/home/amos/MgDB/data/32bp_virus_species.fasta_noAmbiguous_rep_seq.fasta_cdhit80_onebase_10percent_evenly_repeat100_testing"
testPython="test_no_act.py"

medianFile="$modelName-median.pkl"

echo "Start preparing for scatter plot"
# preparation for scatter plot
run_metrics (){
    cd $projectPath
    noise_str="0,0,0,$1"
    python $testPython --metrics --checkpoint $2 \
        --data $testFile --batch-size 100 \
        --noise $noise_str --output 10_100_fp.csv --enc mu \
        --distance_type $3 --seed 20240320 --use-median --median-file $medianFile
}

divergences=(0.05 0.1 0.15 0.2)
distance_types=("euclidean" "cosine" "hamming")
declare -A distance_types_short=(
    [euclidean]="fp"
    [cosine]="cos"
    [hamming]="hm"
)

for divergence in "${divergences[@]}"; do
    for distance_type in "${distance_types[@]}"; do
        short_name=${distance_types_short[$distance_type]}
        run_metrics $divergence $modelPath $distance_type > "${projectPath}/result/100_${divergence}_${short_name}.csv"
    done
done

generate_noise (){
  cd $projectPath
  noise_str="$1,$2,$3,$4"
  python $testPython --generate --checkpoint $5 \
    --data $testFile --batch-size 100 \
    --noise $noise_str --output 10_100_fp.csv --enc mu \
    --distance_type euclidean --seed 20240320 --use-median --median-file $medianFile
}

perform_generation() {
  drop=$1
  add=$2
  sub=$3
  mix=$4
  generate_noise $drop $add $sub $mix $modelPath > $projectPath/result/"$drop,$add,$sub,$mix"_noisy.csv
}

# perform_generation 0   0   0   0.03
perform_generation 0   0   0   0.05
perform_generation 0   0   0   0.1
perform_generation 0   0   0   0.15
perform_generation 0   0   0   0.2

echo "Start preparing for tsne plot"
# preparation for tsne plot
run_embedding (){
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

python scatter_plot.py
python tsne_plot.py --metrics "cosine,hamming" \
    --use-median --median-file "${projectPath}/${modelName}-median.pkl" \
    --save-file "${modelName}-tsne.png"