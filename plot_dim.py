import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seq_embeddings = pickle.load(open('all_embeddings.pkl', 'rb'))
embeddings = np.array([i['embedding_float'][:32] for i  in seq_embeddings])

data = np.transpose(embeddings)[:10,:10000]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Number of bins
bins = 20

# Prepare and plot histogram data
for i, d in enumerate(data):
    # Calculate histogram
    hist, bin_edges = np.histogram(d, bins=bins)
    
    # Calculate bin widths and centers
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plotting each histogram with its unique bin edges
    ax.bar(bin_centers, hist, zs=i, zdir='y', alpha=0.8, width=bin_width, edgecolor='k')

ax.set_xlabel('Value')
ax.set_ylabel('Embedding dimension')
ax.set_zlabel('Density')
ax.set_yticks(np.arange(len(data)))
plt.title('Histogram of each embedding dimension', fontsize=15)
plt.figtext(0.5, 0.01, f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std(axis=-1).mean():.4f}", ha='center', fontsize=15)
plt.savefig('histogram_dim.png')

data = embeddings[:10]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Number of bins
bins = 20

# Prepare and plot histogram data
for i, d in enumerate(data):
    # Calculate histogram
    hist, bin_edges = np.histogram(d, bins=bins)
    
    # Calculate bin widths and centers
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plotting each histogram with its unique bin edges
    ax.bar(bin_centers, hist, zs=i, zdir='y', alpha=0.8, width=bin_width, edgecolor='k')

ax.set_xlabel('Value')
ax.set_ylabel('Embedding Index')
ax.set_zlabel('Density')
ax.set_yticks(np.arange(len(data)))
plt.title('Histogram of Embeddings (Apply quant loss)', fontsize=15)
plt.figtext(0.5, 0.01, f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std(axis=-1).mean():.4f}", ha='center', fontsize=15)
plt.savefig('histogram_z.png')