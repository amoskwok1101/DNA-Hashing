import concurrent.futures
def global_alignment(seq1, seq2, match_score=1, mismatch_penalty=0, gap_penalty=-1):
    seq1 = seq1.split()
    seq2 = seq2.split()

    # Initialize the score matrix
    m, n = len(seq1), len(seq2)
    score_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize the traceback matrix
    traceback_matrix = [[None for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill in the first row and first column of the score matrix
    for i in range(1, m + 1):
        score_matrix[i][0] = i * gap_penalty
        traceback_matrix[i][0] = 'U'  # Up
    for j in range(1, n + 1):
        score_matrix[0][j] = j * gap_penalty
        traceback_matrix[0][j] = 'L'  # Left

    # Fill in the rest of the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty

            max_score = max(match, delete, insert)

            score_matrix[i][j] = max_score
            if max_score == match:
                traceback_matrix[i][j] = 'D'  # Diagonal
            elif max_score == delete:
                traceback_matrix[i][j] = 'U'  # Up
            else:
                traceback_matrix[i][j] = 'L'  # Left

    # Trace back from the bottom right corner to get the alignment
    align1, align2 = '', ''
    i, j = m, n
    while i > 0 or j > 0:
        if traceback_matrix[i][j] == 'D':
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 'U':
            align1 = seq1[i - 1] + align1
            align2 = '-' + align2
            i -= 1
        else:  # 'L'
            align1 = '-' + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    return align1, align2, score_matrix[m][n]
# original_seq = "A G A C A T C G G C G G G C A C A T C C T T G T A G A T G C A 0"
# noisy_seq = "C C T G G C A G A C A T C G G C G G G C A C A T C C T T G T A G"
# print(1 - global_alignment(original_seq, noisy_seq)[2]/len(original_seq.split() ))
# print(global_alignment(original_seq, noisy_seq))
# print(f'seq length: {len(original_seq.split())}')
# original_seq = "A C A T A G A G C G G C A C C T G C C A G C C G T G A T T G A G C C A G G G C G A A A G C G C G A G C C T G A C C T T G C C G A C G A T C T G C G T C C A G A G C T G C A G G G T C G C G G C G G T G T"
# noisy_seq = "A C A T A G A C C G G A A C C T G C C A G C C G T G A T T G A G C C A G G G C G A A A G C G C G A G C C T G A C C T T G C C G A C G A T C T G C G T C C A G A G C T G C A G G G T C G C G G C G G T G T"
# print(1 - global_alignment(original_seq, noisy_seq)[2]/100)
# original_seq = "T A A T C C G G G A G C G C A C G C T T T G A C C A C A T C A A T A A A A A C C A T T C T G A C C G C C G C C C A C T G G G G A"
# original_seq = "1 2 3 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
# noisy_seq1 = "1 2 3 0 1 2 3 4 5 6 7 2 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
# noisy_seq2 = "1 2 3 0 1 2 3 4 5 6 3 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
# noisy_seq3 = "1 2 3 0 1 2 3 4 5 1 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
# list_seq2 = [noisy_seq1, noisy_seq2, noisy_seq3]*1000
# results = []
# import time
# start = time.time()
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     # Prepare a list of future results
#     futures = [executor.submit(global_alignment, original_seq, seq2, 1, 0, 0) for seq2 in list_seq2]
#     for future in concurrent.futures.as_completed(futures):
#         results.append(future.result())
#     end = time.time()
#     print(f"Time taken: {end - start} seconds")
# print(len(original_seq.split()))
# print(results)
# # # Example usage:
# original_seq = "T A A T C C G G G A G C G C A C G C T T T G A C C A C A T C A A T A A A A A C C A T T C T G A C C G C C G C C C A C T G G G G A"
# noisy_seq1 = "T C A A T C C C G G G A C G C G C C A A C T G C T C T T T G A C C A A C C A T G C A A T A A A A A C C C A T T T C T G A C C G C"
# noisy_seq2 = "T A C A A T C C G G G T A G C G A C A C G C T T T T G G A C C C A C A A T C A A T A A A A A C C A T C T C C T T G A A C C G G C"
# noisy_seq3 = "T A A A T C C G G C G A G C G C G A C G C A T G T T T G A C C A A C C A T C A A T A A A C A G A C C A T T T G C T G G A C C G C"
# # original_seq = 'A T C G'
# # noisy_seq1 = 'A T T G'
# # noisy_seq2 = 'A T C G'
# # noisy_seq3 = 'C T A T'
# # Perform global alignment for each noisy sequence and store the score
# for noisy_seq in [noisy_seq1, noisy_seq2, noisy_seq3]:
#     import time
#     start = time.time()
#     seq1_align, seq2_align, score = global_alignment(original_seq, noisy_seq)
#     end = time.time()
#     print(seq1_align)
#     print(seq2_align)
#     print(score)
#     print(f"Time taken: {end - start} seconds")
