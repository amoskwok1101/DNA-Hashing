import numpy as np
import pickle
from tqdm import tqdm
from global_seq_align import global_alignment
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

noisy_list = [0.05, 0.1, 0.15, 0.2]

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
    # print(len(processed_data))
    return np.array(processed_data)

# read original and noisy sequences
# run global alignment
all_scores = [] # (4, 10000)
all_seqs = []
for noisy in noisy_list:
    with open(f'0,0,0,{noisy}_noisy.csv', 'r') as f:
        lines = f.read().splitlines()
    
    is_next_line_original = False
    original = ''
    noisy = []
    scores = []
    seqs = []
    is_next_line_noisy = False
    # for line in lines: # use tqdm
    for line in tqdm(lines, total=len(lines)):
        if line == 'Original':
            is_next_line_noisy = False
            is_next_line_original = True
            # print(len(noisy))
            for noisy_seq in noisy:
                _, _, score = global_alignment(original, noisy_seq)
                scores.append(score)
                all_seqs.append([original.replace(' ',''), noisy_seq.replace(' ','')])
            # print(len(scores))
            noisy = []
            continue
        elif line =='Noisy':
            is_next_line_noisy = True
            continue
        if is_next_line_original:
            original = line
            is_next_line_original = False
        elif is_next_line_noisy:
            noisy.append(line)
    
    for noisy_seq in noisy:
        _, _, score = global_alignment(original, noisy_seq)
        scores.append(score)
        all_seqs.append([original.replace(' ',''), noisy_seq.replace(' ','')])
    # print(len(scores))
    all_scores.append(scores)
    # all_seqs.append(seqs)

with open(f'all_noisy_scores.pkl', 'wb') as f:
    pickle.dump(all_scores, f)
all_seqs = np.array(all_seqs)
# with open(f'all_noisy_scores.pkl', 'rb') as f:
#     all_scores = pickle.load(f)
print(all_seqs.shape)
all_scores = np.array(all_scores)
all_scores = 1 - all_scores / len(all_seqs[0,0])
print(all_scores.shape)

fp = np.array([load_and_process_text(f"100_{noisy}_fp.csv") for noisy in noisy_list])
fp_with_scores = np.stack((all_scores, fp), axis=-1)
cos = np.array([load_and_process_text(f"100_{noisy}_cos.csv") for noisy in noisy_list])
cos_with_scores = np.stack((all_scores, cos), axis=-1)
hm = np.array([load_and_process_text(f"100_{noisy}_hm.csv") for noisy in noisy_list])
hm_with_scores = np.stack((all_scores, hm), axis=-1)
# cos_bin = np.array([load_and_process_text(f"100_{noisy}_cos_bin.csv") for noisy in noisy_list])
# cos_bin_with_scores = np.stack((all_scores, cos_bin), axis=-1)

# Define colors for each of the 4 series in the datasets
colors = ['red', 'green', 'orange', 'purple']  # Colors for the series

# Create a figure with three subplots, arranged vertically
plt.figure(figsize=(10, 15))



def custom_scatterplot(i, by_scores, title):
    plt.subplot(1, 3, i)
    for j in range(4):
        plt.scatter(by_scores[j, :, 0], by_scores[j, :, 1], color=colors[j], alpha=0.5, label=noisy_list[j])
        
    # Perform linear regression
    X = by_scores[:, :, 0].reshape(-1, 1)
    print(X.shape)
    y = by_scores[:, :, 1].reshape(-1, 1)
    print(y.shape)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
        
    # Plot the regression line
    # plt.plot(X, y_pred, color='black', alpha=0.5)
    
    # Display R2 value
    plt.text(0.05, 0.9, f'R²: {r2:.2f}', transform=plt.gca().transAxes,
                color='black', fontsize=30, verticalalignment='top')
        
    plt.xlabel('Score')
    plt.ylabel('Distance')
    plt.title(title)
    plt.legend()

custom_scatterplot(1, fp_with_scores, 'Euclidean distance')
custom_scatterplot(2, cos_with_scores, 'Cosine distance')
custom_scatterplot(3, hm_with_scores, 'Hamming distance')
plt.tight_layout()
# plt.savefig(args.save_file)
plt.savefig('scatter_plot.png')
# import pandas as pd
# df = pd.DataFrame(columns=['divergence', 'distance'])
# for i in range(len(colors)):
#     df_temp = pd.DataFrame({'divergence': fp_with_scores[i, :, 0], 'distance': fp_with_scores[i, :, 1]})
#     df = pd.concat([df, df_temp])
# # add new column to df named 'divergence group'
# df['divergence group'] = pd.cut(df['divergence'], bins=[0, 0.05, 0.1, 0.15, 0.2], labels=['0.05', '0.1', '0.15', '0.2'])
# df.to_csv('fp.csv', index=False)
# df_seqs = pd.DataFrame(all_seqs, columns=['original', 'noisy'])
# df_seqs.to_csv('seqs.csv', index=False)
# import seaborn as sns
# # Create a box plot
# plt.figure(figsize=(10,15))  # Set the figure size (optional)
# boxplot = df.boxplot(by='divergence group', column=['distance'], grid=False)
# plt.title('Box Plot of Distance by Divergence')  # Add a title to the plot
# plt.suptitle('')  # Suppress the default title to clean up the plot
# plt.xlabel('Divergence')  # Label the x-axis
# plt.ylabel('Distance')  # Label the y-axis
# plt.show()
# plt.savefig('box_plot_real_div.png')

# plt.figure(figsize=(15,15))
# sns.violinplot(x='divergence group', y='distance', data=df, palette='Blues')
# plt.title('Violin Plot of Distance by Divergence')
# plt.xlabel('Divergence')
# plt.ylabel('Distance')
# plt.show()
# plt.savefig('violin_plot_real_div.png')
# # df = df[df['divergence']!=0]
# df_sort = df.sort_values(by='divergence', ascending=True)
# df_sort = df_sort[df_sort['divergence']!=0]
# divergence = df_sort['divergence'].unique()
# res = []
# for i in divergence:
#     # get the max distance of i
#     max_distance = df_sort[df_sort['divergence'] == i]['distance'].max()
#     # get the rows of min distance group by each divergence
#     min_distances = df_sort.groupby('divergence')['distance'].min()[(divergence > i)]
    
#     border = min_distances[min_distances>max_distance]
#     if border.empty:
#         border = np.NaN
#     else:
#         border = border.index[0]
#     # get the min divergence of rows
#     # min_divergence = rows['divergence'].min()
#     # # get the difference in divergence
#     diff = border - i
#     res.append([i, border, diff])

# import matplotlib.pyplot as plt

# plt.figure()  # Create a new figure
# plt.title('Divergence Border')
# plt.xlabel('divergence')
# plt.ylabel('border (divergence)')
# plt.plot(np.array(res)[:,0],np.array(res)[:,1])
# plt.xlim(0, np.max(np.array(res)[:,0]))
# plt.savefig('divergence_border.png')

# plt.figure() 
# plt.title('Divergence Border Difference')
# plt.xlabel('divergence')
# plt.ylabel('border difference')
# plt.plot(np.array(res)[:,0],np.array(res)[:,2])
# plt.xlim(0, np.max(np.array(res)[:,0]))
# plt.savefig('divergence_diff.png')

# df = pd.read_csv('fp.csv')
# df_seq = pd.read_csv('seqs.csv')
# df_seq_score = pd.concat([df, df_seqs], axis=1)
# df_seq_score = df_seq_score[df_seq_score['divergence']!=0]
# df_drop_pad = df_seq_score[~df_seq_score['noisy'].str.contains('<pad>')]
# plt.figure(figsize=(10,15))  # Set the figure size (optional)
# boxplot = df_drop_pad.boxplot(by='divergence group', column=['distance'], grid=False)
# plt.title('Box Plot of Distance by Divergence')  # Add a title to the plot
# plt.suptitle('')  # Suppress the default title to clean up the plot
# plt.xlabel('Divergence')  # Label the x-axis
# plt.ylabel('Distance')  # Label the y-axis
# plt.show()
# plt.savefig('box_plot_real_div_no_pad.png')

# noisy_list = [0, 0.05, 0.1, 0.15, 0.2]
# # if fastafile exists, remove it
# import os
# if os.path.exists("min_max_ori_noisy.fasta"):
#     os.remove("min_max_ori_noisy.fasta")
# for i in range(len(noisy_list)-1):
#     within_range = df_seq_score[(df_seq_score['divergence'] <= noisy_list[i+1]) &(df_seq_score['divergence'] > noisy_list[i])].sort_values(by='distance', ascending=True)
#     min_row = within_range.iloc[0]
#     max_row = within_range.iloc[-1]
#     # print(min_row)
#     # print(max_row['divergence'])
#     fasta_file = "min_max_ori_noisy.fasta"

#     with open(fasta_file, "a") as file:
#         file.write(">max_{:.2f}_div_{:.6f}_euc_{:.6f}_ori\n".format(noisy_list[i+1],max_row["divergence"], max_row["distance"]))
#         file.write(max_row["original"] + "\n")
#         file.write(">max_{:.2f}_div_{:.6f}_euc_{:.6f}_noisy\n".format(noisy_list[i+1],max_row["divergence"], max_row["distance"]))
#         file.write(max_row["noisy"] + "\n")
#         file.write(">min_{:.2f}_div_{:.6f}_euc_{:.6f}_ori\n".format(noisy_list[i+1],min_row["divergence"], min_row["distance"]))
#         file.write(min_row["original"] + "\n")
#         file.write(">min_{:.2f}_div_{:.6f}_euc_{:.6f}_noisy\n".format(noisy_list[i+1],min_row["divergence"], min_row["distance"]))
#         file.write(min_row["noisy"] + "\n")

#     print("Content appended to FASTA file:", fasta_file)