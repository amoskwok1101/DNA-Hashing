import matplotlib.pyplot as plt
import numpy as np
import csv

prefix = '/home/amos/MgDB/result/'
f = prefix + '100_0.02_fp.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d1 = d

f = prefix + '100_0.05_fp.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d2 = d    

f = prefix + '100_0.1_fp.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d3 = d  

f = prefix + '100_0.2_fp.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d4 = d  

fp = [d1, d2, d3, d4]

f = prefix + '100_0.02_cos.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d1 = d

f = prefix + '100_0.05_cos.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d2 = d    

f = prefix + '100_0.1_cos.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d3 = d  

f = prefix + '100_0.2_cos.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d4 = d  

cos = [d1, d2, d3, d4]

f = prefix + '100_0.02_hm.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d1 = d

f = prefix + '100_0.05_hm.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d2 = d    

f = prefix + '100_0.1_hm.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d3 = d  

f = prefix + '100_0.2_hm.csv'
with open(f) as f:
    reader = csv.reader(f)
    d = np.array(list(reader))
    d = d[:,0:-1].flatten().astype(float)
    d4 = d  

hm = [d1, d2, d3, d4]

# fake data
fs = 20  # fontsize
pos = [1, 2, 3, 4]
labels = ['0.02', '0.05', '0.1', '0.2']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

axs[0].boxplot(fp, sym = '', widths=0.7, vert=True, labels=labels)
axs[0].set_title('Euclidean distance', fontsize=fs)
axs[0].set_xticklabels(labels)
axs[0].set_xlabel('Divergence')

axs[1].boxplot(cos, sym = '', widths=0.7, vert=True, labels=labels)
axs[1].set_title('Cosine distance', fontsize=fs)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel('Divergence')

axs[2].boxplot(hm, sym = '', widths=0.7, vert=True, labels=labels)
axs[2].set_title('Hamming distance', fontsize=fs)
axs[2].set_xticklabels(labels)
axs[2].set_xlabel('Divergence')

plt.savefig('/home/amos/MgDB/result/test.jpg')

