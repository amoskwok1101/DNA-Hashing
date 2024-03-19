#!/bin/sh

export TOKENIZERS_PARALLELISM=true
modelPath=/container_data/ytye/SRA/codes/DAAE/checkpoints/dae32bits_att/

test_fp_func (){
	cd /container_data/ytye/SRA/codes/DAAE
#testFile=./data/dna/10_100_64.txt
    noise_str=0,0,0,$1
  	testFile=../../real/virus_all.fasta_noAmbiguous_cdhit_onebase_test_repeat100
  	python test.py --metrics --checkpoint $2 \
    		--data $testFile --batch-size 100 \
    		--noise $noise_str --output 10_100_fp.csv --enc mu \
			--distance_type euclidean
}

test_hm_func (){
  	cd /container_data/ytye/SRA/codes/DAAE
#testFile=./data/dna/10_100_64.txt
  	testFile=../../real/virus_all.fasta_noAmbiguous_cdhit_onebase_test_repeat100
	noise_str=0,0,0,$1
  	python test.py --metrics --checkpoint $2 \
    	--data $testFile --batch-size 100 \
    	--noise $noise_str --output 10_100_fp.csv --enc mu \
    	--distance_type hamming
}

test_cos_func (){
        cd /container_data/ytye/SRA/codes/DAAE
#testFile=./data/dna/10_100_64.txt
        testFile=../../real/virus_all.fasta_noAmbiguous_cdhit_onebase_test_repeat100
		noise_str=0,0,0,$1
        python test.py --metrics --checkpoint $2 \
        --data $testFile --batch-size 100 \
        --noise $noise_str --output 10_100_fp.csv --enc mu \
        --distance_type cosine
}

divergence=0.02
test_fp_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_fp.csv"
divergence=0.05
test_fp_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_fp.csv"
divergence=0.1
test_fp_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_fp.csv"
divergence=0.2
test_fp_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_fp.csv"

divergence=0.02
test_cos_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_cos.csv"
divergence=0.05
test_cos_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_cos.csv"
divergence=0.1
test_cos_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_cos.csv"
divergence=0.2
test_cos_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_cos.csv"

divergence=0.02
test_hm_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_hm.csv"
divergence=0.05
test_hm_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_hm.csv"
divergence=0.1
test_hm_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_hm.csv"
divergence=0.2
test_hm_func $divergence $modelPath > /container_data/ytye/SRA/codes/DAAE/result/100_$divergence"_hm.csv"

cd /container_data/ytye/SRA/codes/DAAE/result/
python box_plot.py
