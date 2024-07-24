# MgDB

## Preparation

use environment.yml to replicate the conda environment, which include suitable python version and accelerate library for ddp training 
```
conda env create -f environment.yml
```

## Modify runTrain.sh for your model training

```
trainFile=/home/amos/MgDB_amos/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_10000_testing
validFile=/home/amos/MgDB_amos/data/virus_species_v2.fasta_rep_seq_cdhit80.fasta_onebase_10percent_evenly_10000_testing
modelPath=/home/amos/MgDB_amos/checkpoints/dae_cosine_2
```
**if the data is in onebase format (spaces in between each base, e.g.'A T G C'), modify output format of load_sent in utils.py as follow:**
```
# data in onebase format
sents.append(line.split())
# otherwise
sents.append(list(line.strip()))
```

To run training with more than 1 gpu, modify the --num_processes as desired
```
python -m accelerate.commands.launch --num_processes=1  train.py --train $trainFile --valid $validFile
```

## Run Test after training

```
# Plot both scatter plot and tsne plot
./runPlot.sh
# Plot only scatter plot
./runScatterplot.sh
# Plot only tsne plot
./runTsne.sh
# Run retreival test
./runTestRetrieval.sh
```

**For plot graphs, please use data with "_evenly_repeat100_testing" extension**
**For retrieval test, please use data with "_evenly_10000_testing" extension**

