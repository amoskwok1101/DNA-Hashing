# Preprocess FASTA to training data
**Raw species files are situated on Prog2 Account, so please perform data preprocess (except cdhit) on Prog2 Account**

**species-small.txt & virus.txt** are the fasta file that I usually use to produce those small dataset (e.g. 2.46GB dataset), while files with filename in **species-large.txt** take much longer time to preprocess.

## 1. Preparation
Dowload [seqkit](https://bioinf.shenwei.me/seqkit/download/) and [mmseq](https://github.com/soedinglab/MMseqs2)
```
conda install -c bioconda seqkit
conda install -c conda-forge -c bioconda mmseqs2
```
## 2. Using sliding window to slice raw seq to 100/64/32bp files
Modify below parameter in preprocess.py for your use
```
INPUT_FOLDER_PATH = "/home/d24h_prog2/isaac-wu/dnabert2-inputs-preprocess/inputs"
OUTPUT_FOLDER_PATH = "/home/d24h_prog2/amos/virus-species-output-32"
FINISHED_SPECIES_PATH = "/home/d24h_prog2/amos/large.txt"
K_MER_LENGTH = 32
```
**INPUT_FOLDER_PATH**: file path for raw sequences

/home/d24h_prog2/isaac-wu/dnabert2-inputs-preprocess/virus-inputs: virus fasta files

/home/d24h_prog2/isaac-wu/dnabert2-inputs-preprocess/inputs: other species fasta files

**OUTPUT_FOLDER_PATH**: your output file path

**FINISHED_SPECIES_PATH**: record the file path that finished processing (in case the code terminate in between)

**K_MER_LENGTH**: your desired bp length

```
python preprocess.py
```
to get n-bp sequences fasta file

## 3. Run removeAmbiguous on each fasta file
Run ./removeAmbiguous to reduce fasta file size before running mmseq or cdhit

## 4. Run mmseq 2/3 times on individual/merged files to reduce file size
```
# easy-linclust suitable for large fasta file
mmseqs easy-cluster/easy-linclust $input $output $tmp_folder --min-seq-id 0.8 --threads 32
```
For $tmp_folder, just create any empty directory for mmseq to output intermediate files.

The output will be:

($output)_all_seqs.fasta

($output)_cluster.tsv

($output)_rep_seq.fasta **we only need this file**

**Note: depends on the file size, you may run mmseq on individual, merged or splitted file (Refer to appendix for split using seqkit)**

## 5. Run cdhit 0.8 for better filtering
```
cd-hit/cd-hit-est -i $input -c 0.8 -o $output -M 0 -T 0
```
-c  sequence identity threshold, default 0.9 (output will have <0.9 similarity);
-M	memory limit (in MB) for the program, default 800; 0 for unlimitted;
-T	number of threads, default 1; with 0, all CPUs will be used

The output will be:

($output) **we only need this**

($output).clstr 

# Appendix

## Split large fasta using seqkit
```
seqkit split2 -p 4 -j 4 -O $output_dir $input_file_path
```
-p, --by-part int   split sequences into N parts

-j, --threads int   number of CPUs

Suppose your $input_file_path: test.fasta
The output will be:
test.part_00{1-n}.fasta

## Use Carlos account to run cdhit
cdhit takes a lot more time to process and Prog2 account will terminate job for unknown reason, so I suggest moving your files to Carlos account to run cdhit
