import matplotlib.pyplot as plt
import numpy as np

prefix = '/home/amos/MgDB/result/'
fp_files = ['100_0.02_fp.csv', '100_0.05_fp.csv', '100_0.1_fp.csv', '100_0.2_fp.csv']
cos_files = ['100_0.02_cos.csv', '100_0.05_cos.csv', '100_0.1_cos.csv', '100_0.2_cos.csv']
hm_files = ['100_0.02_hm.csv', '100_0.05_hm.csv', '100_0.1_hm.csv', '100_0.2_hm.csv']
cos_bin_files = ['100_0.02_cos_bin.csv', '100_0.05_cos_bin.csv', '100_0.1_cos_bin.csv', '100_0.2_cos_bin.csv']
def load_and_process_text(filename):
    processed_data = []
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            numbers = [num.strip() for num in line.split(',') if num.strip()]
            try:
                processed_data.extend([float(num) for num in numbers])
            except ValueError as e:
                print(f"ValueError on line {line_number}: {e}")
                print(f"Problematic data: {numbers}")
    print(len(processed_data))
    return np.array(processed_data)

fp = [load_and_process_text(prefix + f) for f in fp_files]
cos = [load_and_process_text(prefix + f) for f in cos_files]
hm = [load_and_process_text(prefix + f) for f in hm_files]
cos_bin = [load_and_process_text(prefix + f) for f in cos_bin_files]
fs = 20
# print(np.array(fp).shape)
labels = ['0.02', '0.05', '0.1', '0.2']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

axs[0].boxplot(fp, sym='', widths=0.7, vert=True, labels=labels)
axs[0].set_title('Euclidean distance', fontsize=fs)
axs[0].set_xticklabels(labels)
axs[0].set_xlabel('Divergence')

axs[1].boxplot(cos, sym='', widths=0.7, vert=True, labels=labels)
axs[1].set_title('Cosine distance', fontsize=fs)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel('Divergence')

axs[2].boxplot(hm, sym='', widths=0.7, vert=True, labels=labels)
axs[2].set_title('Hamming distance', fontsize=fs)
axs[2].set_xticklabels(labels)
axs[2].set_xlabel('Divergence')

# axs[3].boxplot(cos_bin, sym='', widths=0.7, vert=True, labels=labels)
# axs[3].set_title('Cosine distance (binary)', fontsize=fs)
# axs[3].set_xticklabels(labels)
# axs[3].set_xlabel('Divergence')

plt.savefig('/home/amos/MgDB/result/box_plot.png')