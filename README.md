# DNA Hashing
This repo include scripts to train and test encoder model to encode DNA sequences in binary codes for fast retrieval

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

for detailed model training arguments, please refer to model_flag.txt

## Submit job to HPC using pbs script

Check node status using
```
pbsnodes -aSj
```

Below is a snippet of the command output
```
                                                        mem       ncpus   nmics   ngpus
vnode           state           njobs   run   susp      f/t        f/t     f/t     f/t   jobs
--------------- --------------- ------ ----- ------ ------------ ------- ------- ------- -------
hpc-gn001       free                 3     3      0      1tb/1tb   20/48     0/0     0/4 99351.hpc-hn001,99393.hpc-hn001,99351.hpc-hn001
hpc-gn002       free                 1     1      0      1tb/1tb   43/48     0/0     0/4 99179.hpc-hn001
hpc-gn003       free                 2     2      0    872gb/1tb   17/48     0/0     0/4 99420.hpc-hn001,99408.hpc-hn001
hpc-gn004       job-busy             2     2      0    872gb/1tb    0/48     0/0     0/4 97043.hpc-hn001,97520.hpc-hn001
```
**For model training, submit job to gpuq1. Mainly focus on ncpus and ngpus f(ree)/t(otal) when deciding amount of resources for your job(s). Your job will be allocated to a free node by the system**

For instruction on writting pbs script, refer to **train.pbs**.

To submit job, run 
```
qsub $your_pbs_script
```

You should then see your job status with
```
qstat
```

```
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
99420.hpc-hn001   dna_hash_ladder  carloschau_prog2  33:12:55 R gpuq1           
99421.hpc-hn001   dna_hash_ladder  carloschau_prog2  00:31:52 R gpuq1           
99422.hpc-hn001   dna_hash_ladder  carloschau_prog2         0 Q gpuq1           
```
if your job is running S(tatus) will be R, Q means your job is on queue.

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

**For plotting graphs, please use data with "_evenly_repeat100_testing" extension**

**For retrieval test, please use data with "_evenly_10000_testing" extension**

