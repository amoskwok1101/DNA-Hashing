from pathlib import Path
import re

INPUT_FOLDER_PATH = "/home/d24h_prog2/isaac-wu/dnabert2-inputs-preprocess/inputs"
OUTPUT_FOLDER_PATH = "/home/d24h_prog2/amos/virus-species-output-32"
FINISHED_SPECIES_PATH = "/home/d24h_prog2/amos/large.txt"
K_MER_LENGTH = 32

def main(input_folder, output_folder, finished_species_filepath):
    file_list = get_file_paths(input_folder)
    # printf(file_list)
    printf([str(f) for f in file_list])
    printf(f"len of file list: {len(file_list)}")
    finished_species = get_finished_species(finished_species_filepath)
    printf(finished_species)
    printf(f"len of finished species: {len(finished_species)}")
    for filepath in file_list:
        # printf(filepath)
        with open(filepath, "r") as f:
            species_name = get_species_name(filepath)
            # printf(species_name)
            if species_name not in finished_species:
                process_file(f, species_name, output_folder)
                add_species_to_file(finished_species_filepath, species_name)

def get_file_paths(folder_path: str):
    input_dir = Path(folder_path)
    return [
        f
        for f in input_dir.glob('**/*') 
        if f.is_file()
    ]

def printf(message):
    print(message, flush=True)

def get_finished_species(filepath):
    species = []
    with open(filepath, "r") as f:
        for line in f:
            s = line.replace("\n", "")
            if s != "":
                species.append(s)
    return species

def get_species_name(filepath):
    return filepath.parent.name.replace(" ", "_")

def process_file(input_file, species_name, output_folder_path):
    tmp_ncbi_id = ""
    tmp_seq = ""
    output_file_path = f"{output_folder_path}/{species_name}.fasta"
    printf(output_file_path)
    with open(output_file_path, "w") as output_file:
        for line in input_file:
            line = line.replace("\n", "")
            if line.startswith(">"):
                if tmp_ncbi_id != "":
                    process_seq(tmp_ncbi_id, tmp_seq, output_file)
                    tmp_seq = ""
                tmp_ncbi_id = get_ncbi_id(line)
            else:
                tmp_seq += line
        process_seq(tmp_ncbi_id, tmp_seq, output_file)

def process_seq(ncbi_id, seq, output_file):
    tmp_is_atcg_only_previous = False
    for i in range(len(seq) - K_MER_LENGTH + 1):
        trimmed_seq = seq[i:i+K_MER_LENGTH]
        if tmp_is_atcg_only_previous:
            is_atcg_only_current = is_atcg_only_last_index(trimmed_seq)
        else:
            is_atcg_only_current = is_atcg_only(trimmed_seq)  
        tmp_is_atcg_only_previous = is_atcg_only_current
        is_atcg_only_char = "Y" if is_atcg_only_current else "N"
        metadata = f">{ncbi_id}|{i}|{is_atcg_only_char}"
        output_file.write(metadata + "\n" + trimmed_seq + "\n")

def is_atcg_only_last_index(seq):
    return seq[-1] in ["A", "T", "C", "G"]

def is_atcg_only(seq):
    return bool(re.match('^[ATCG]+$', seq))
    
def get_ncbi_id(line):
    return line.split()[0][1:]

def add_species_to_file(filepath, species_name):
    with open(filepath, "a") as f:
        f.write(species_name + "\n")

if __name__ == "__main__":
    main(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, FINISHED_SPECIES_PATH)
