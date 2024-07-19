import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch 
import seaborn as sns

with open('radius_distribution.pkl', 'rb') as f:
    radius_distribution = pickle.load(f)

with open('candidates_distribution.pkl', 'rb') as f:
    candidates_distribution = pickle.load(f)
    
# f"hamming_{noise_level}", f"euc_{noise_level}"

def hist_plot(data_1, data_2, title_1, title_2, xlabel, save_file):
    data_1 = [int(item) for item in data_1]
    data_2 = [int(item) for item in data_2]
    
    # Calculate statistics for both datasets
    stats_1 = {
        'mean': np.mean(data_1),
        'min': np.min(data_1),
        'max': np.max(data_1),
        'median': np.median(data_1)
    }
    
    stats_2 = {
        'mean': np.mean(data_2),
        'min': np.min(data_2),
        'max': np.max(data_2),
        'median': np.median(data_2)
    }
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Data and titles in a list for looping
    datasets = [(data_1, title_1, 'plum'), (data_2, title_2, 'lightblue')]

    # Determine the minimum and maximum values for the x-axis
    min_val = min(min(data_1), min(data_2))
    max_val = max(max(data_1), max(data_2))
    
    # Width of each bar
    bar_width = 0.25

    # Compute histogram positions
    r = np.arange(min_val, max_val + 1)

    # Plot histograms side by side
    for i, (data, title, color) in enumerate(datasets):
        hist, edges = np.histogram(data, bins=np.arange(min_val, max_val + 2) - 0.5)
        bar_positions = r + i * bar_width
        ax.bar(bar_positions, hist, width=bar_width, edgecolor='black', color=color, alpha=0.7, label=title)

        # Annotate bars
        for pos, h in zip(bar_positions, hist):
            ax.text(pos, h + 0.05 * max(hist), '%d' % int(h), ha='center', va='bottom')

        # Add text with total number of frequencies in the upper right corner
        total_frequency = len(data)
        ax.text(0.9, 0.95 - i * 0.05, f'Total {title}: {total_frequency}', ha='right', va='top', transform=ax.transAxes)

    # Adjust xticks to center them between the grouped bars
    ax.set_xticks(r + bar_width / 2)
    ax.set_xticklabels(range(min_val, max_val + 1))

    # Set the title and labels
    ax.set_title(f'Distribution of {title_1} and {title_2}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    
    # Add a legend
    ax.legend()

    # Create a table with the statistics
    table_data = [
        [title_1, f"{stats_1['mean']:.2f}", f"{stats_1['min']:.2f}", f"{stats_1['max']:.2f}", f"{stats_1['median']:.2f}"],
        [title_2, f"{stats_2['mean']:.2f}", f"{stats_2['min']:.2f}", f"{stats_2['max']:.2f}", f"{stats_2['median']:.2f}"]
    ]
    
    column_labels = [' ', 'Mean', 'Min', 'Max', 'Median']
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='bottom', cellLoc='center', bbox=[0.0, -0.35, 1.0, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.28)  # Adjust bottom to make space for the table
    plt.savefig(save_file)

def kde_plot(data_1, data_2, title_1, title_2, xlabel, save_file):
    # Calculate statistics for both datasets
    stats_1 = {
        'mean': np.mean(data_1),
        'min': np.min(data_1),
        'max': np.max(data_1),
        'median': np.median(data_1)
    }
    
    stats_2 = {
        'mean': np.mean(data_2),
        'min': np.min(data_2),
        'max': np.max(data_2),
        'median': np.median(data_2)
    }

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot KDE for both datasets
    sns.kdeplot(data_1, fill=True, color='plum', label=title_1, ax=ax)
    sns.kdeplot(data_2, fill=True, color='lightblue', label=title_2, ax=ax)
    
    # Draw median lines and annotate them
    median_1 = stats_1['median']
    median_2 = stats_2['median']
    
    ax.axvline(median_1, color='purple', linestyle='--')
    ax.text(median_1, ax.get_ylim()[0]-0.1, f'{median_1:.2f}', color='purple', ha='center', va='bottom', transform=ax.get_xaxis_transform())
    
    ax.axvline(median_2, color='blue', linestyle='--')
    ax.text(median_2, ax.get_ylim()[0]-0.1, f'{median_2:.2f}', color='blue', ha='center', va='bottom', transform=ax.get_xaxis_transform())
    
    for i, data in enumerate([data_1, data_2]):
        title = title_1 if i == 0 else title_2
        total_frequency = len(data)
        ax.text(0.9, 0.95 - i * 0.05, f'Total {title}: {total_frequency}', ha='right', va='top', transform=ax.transAxes)

    # Set the title and labels
    ax.set_title(f'Distribution of {title_1} and {title_2}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    
    # Add a legend
    ax.legend()

    # Create a table with the statistics
    table_data = [
        [title_1, f"{stats_1['mean']:.2f}", f"{stats_1['min']:.2f}", f"{stats_1['max']:.2f}", f"{stats_1['median']:.2f}"],
        [title_2, f"{stats_2['mean']:.2f}", f"{stats_2['min']:.2f}", f"{stats_2['max']:.2f}", f"{stats_2['median']:.2f}"]
    ]
    
    column_labels = [' ', 'Mean', 'Min', 'Max', 'Median']
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='bottom', cellLoc='center', bbox=[0.0, -0.35, 1.0, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(save_file)

hist_plot(radius_distribution['hamming_0.02'], radius_distribution['hamming_0.05'], '0.02', '0.05', 'Hamming radius', 'hamming_distribution.png')
kde_plot(radius_distribution['euc_0.02'], radius_distribution['euc_0.05'], '0.02', '0.05', 'Euclidean radius', 'euc_distribution.png')

candidates_distribution['hamming_0.02'] = [item.item() for item in candidates_distribution['hamming_0.02']]
candidates_distribution['hamming_0.05'] = [item.item() for item in candidates_distribution['hamming_0.05']]
candidates_distribution['euc_0.02'] = [item.item() for item in candidates_distribution['euc_0.02']]
candidates_distribution['euc_0.05'] = [item.item() for item in candidates_distribution['euc_0.05']]

kde_plot(candidates_distribution['hamming_0.02'], candidates_distribution['hamming_0.05'], '0.02', '0.05', 'Hamming candidates', 'hamming_candidates_distribution.png')
hist_plot(candidates_distribution['euc_0.02'], candidates_distribution['euc_0.05'], '0.02', '0.05', 'Euclidean candidates', 'euc_candidates_distribution.png')