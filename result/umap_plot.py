import numpy as np
import torch
import torch.nn as nn
import pickle
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save-file', type=str, default='umap_cosine_hamming_divergences_grid.png',
                    help='file name of the saved plot')
parser.add_argument('--metrics', type=str, default='cosine,hamming',
                    help='2 metric used for umap choise: (cosine, euclidean, hamming)')
parser.add_argument('--use-median', action='store_true',
                    help='use 32 median for toBinary cutoff')
parser.add_argument('--median-file', metavar='FILE',
                    help='file path of the median')
args = parser.parse_args()

divergences = ['0.05', '0.1', '0.15', '0.2']

def sigmoid(x):
    return nn.Sigmoid()(torch.tensor(x, dtype=torch.float32)).numpy()

def toBinary(code):
    # code = sigmoid(code)
    if args.use_median and args.median_file is not None:
        median = pickle.load(open(args.median_file, 'rb'))
        binary_code = (code > median).astype(int)  # Using threshold of median
    else:
        binary_code = (code > 0.5).astype(int) # Using threshold of 0.5
    return binary_code

# Create the plot with 2 rows and 4 columns for the combined grid
fig, axes = plt.subplots(2, 4, figsize=(20, 10)) # 2 rows (cosine, hamming), 4 columns (divergences)

metrics = args.metrics.split(',')

for col, divergence in enumerate(divergences):
    with open('all_embeddings_[0.0, 0.0, 0.0, '+divergence+'].pkl', 'rb') as f:
        data = pickle.load(f)

    data = np.array(data) # [100, 101, 32]
    data = data.reshape(-1, 32) # [10100, 32]

    for index, metric in enumerate(metrics):
        if metric == 'cosine':
            # Normalize data for cosine distance
            # normalized_data = normalize(data.reshape(-1, 32), axis=1)
            # data = sigmoid(data)
            distances = squareform(pdist(data, 'cosine'))
        
        elif metric == 'hamming':
            binarized_data = toBinary(data)
            distances = squareform(pdist(binarized_data, 'hamming'))
        
        elif metric == 'euclidean':
            # data = sigmoid(data)
            distances = squareform(pdist(data, 'euclidean'))

        # Perform t-SNE
        umap_metric = UMAP(n_components=2, metric='precomputed', random_state=42,init='random')
        embeddings_2d = umap_metric.fit_transform(distances)

        colors = cm.rainbow(np.linspace(0, 1, 100))  # 100 colors for 100 groups
        for i, color in zip(range(100), colors):
            axes[index, col].scatter(embeddings_2d[i*101:(i+1)*101, 0], embeddings_2d[i*101:(i+1)*101, 1], s=15, color=color)

        axes[index, col].set_title(f'{metric.capitalize()}, divergence={divergence}')

plt.tight_layout()
plt.savefig(args.save_file)