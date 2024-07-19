import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

with open('hit_minus_top1_distances.pkl', 'rb') as f:
    hit_minus_top1_distances = pickle.load(f)

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
    fig, ax = plt.subplots(figsize=(7, 5))

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
        ax.text(0.8, 0.95 - i * 0.05, f'Total {title}: {total_frequency}', ha='right', va='top', transform=ax.transAxes)

    # Adjust xticks to center them between the grouped bars
    ax.set_xticks(r + bar_width / 2)
    ax.set_xticklabels(range(min_val, max_val + 1))

    # Set the title and labels
    ax.set_title(f'Distribution of {xlabel} for {title_1} and {title_2}')
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

hit_minus_top1_distances['0.02'] = [int(i) for i in hit_minus_top1_distances['0.02'] if i != -1]
hit_minus_top1_distances['0.05'] = [int(i) for i in hit_minus_top1_distances['0.05'] if i != -1]
hist_plot(hit_minus_top1_distances['0.02'], hit_minus_top1_distances['0.05'], '0.02', '0.05', 'hit - top1 distance', 'hit_minus_top1_distribution.png')
