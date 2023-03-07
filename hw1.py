import numpy as np
from enum import IntEnum
import pandas as pd
import sys

#!/usr/bin/python
__author__ = "Megha Joshi"
__email__ = "megha.joshi@yale.edu"
__copyright__ = "Copyright 2021"
__license__ = "GPL"
__version__ = "1.0.0"

### Usage: python hw1.py -i <input file> -s <score file>
### Example: python hw1.py -i input.txt -s blosum62.txt
### Note: Smith-Waterman Algorithm

import argparse

### This is one way to read in arguments in Python. 
parser = argparse.ArgumentParser(description='Smith-Waterman Algorithm')
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-s', '--score', help='score file', required=True)
parser.add_argument('-o', '--opengap', help='open gap', required=False, default=-2)
parser.add_argument('-e', '--extgap', help='extension gap', required=False, default=-1)
args = parser.parse_args()

#For a given gap penalties, the program should output the best alignment of two sequences.
#The default gap penalties are as follows: opening gap -2, extension gap = -1

# Created dictionary to easily access similarity matrix values
AMINO_ACIDS = "ABCDEFGHIKLMNPQRSTVWXYZ"
simDict = dict([(x[1],x[0]) for x in enumerate(AMINO_ACIDS)])

# (1) Sequences, where the input is shown 
def printSequences(inputFile, f):
    print("-----------\n|Sequences|\n-----------", file=f)
    sequences = open(inputFile).read().splitlines()
    for count, sequence in enumerate(sequences):
        print("sequence" + str(count + 1), file=f)
        print(sequence, file=f)
    return sequences

# (2) Score matrix, where the completed scoring matrix are shown in tab-delimited format (akin to the hand-drawn scoring matrix) 
def printScoreMatrix(matrix, sequence1, sequence2, f):
    print("--------------\n|Score Matrix|\n--------------", file=f)
    seq1 = ' ' + sequence1[:]
    seq2 = ' ' + sequence2[:]
    df = pd.DataFrame(matrix, index = list(seq2), columns = list(seq1))
    print(df.to_csv(sep = '\t'), file=f, end="")

# (3) Best alignment output as well as the alignment score
def printBestLocalAlignment(sequence1, match_string, sequence2, max_score, f):
    print("----------------------\n|Best Local Alignment|\n----------------------", file=f)
    print("Alignment Score:" + str(max_score), file=f)
    print("Alignment Results:", file=f)
    print(sequence1, file=f)
    print(match_string, file=f)
    print(sequence2, file=f)

# Create trace class to track stop, left, up and diagonal. Got this idea from looking at different implementations of SW
class Trace(IntEnum):
    STOP = 0
    LEFT = 2
    UP = 1
    DIAGONAL = 3

"""Create similarity matrix from txt file"""
def createSimilarityMatrix(scoreFile):
    similarity_matrix = np.zeros((23, 23), dtype=int)
    file = open(scoreFile)
    lines = file.readlines()

    for line_num, line in enumerate(lines):
        if line_num != 0:
            nums = line.split()
            for index, num in enumerate(nums):
                if index != 0:
                    similarity_matrix[index-1][line_num-1] = int(num)

    return similarity_matrix

"""Looks up similarity score using defined global dict"""
def findSimilarityScore(letter1, letter2, similarity_matrix):
    index1 = simDict[letter1]
    index2 = simDict[letter2]
    return similarity_matrix[index1][index2]


""" Calculates score matrix using SW algorithm"""
def scoreSW(openGap, extGap, sequence1, sequence2, similarity_matrix, matrix):
    # Initializing variables
    up_gap_penalty = []
    left_gap_penalty = []
    rows = len(sequence1) + 1
    cols = len(sequence2) + 1
    max_score = -1
    max_score_indicies = (-1, -1)
    traceback_matrix = np.zeros((rows, cols), dtype=int) #Creating matrix to keep track of optimal traceback
    direction_matrix = np.zeros((rows, cols), dtype=int)

    for row in range(1, len(sequence1) + 1):
        for col in range(1, len(sequence2) + 1):

            # get directly diagonal by adding blossom score to self
            match_value = findSimilarityScore(sequence2[col - 1], sequence1[row - 1], similarity_matrix)
            diagonal = matrix[row - 1][col - 1] + match_value

            # get all values directly up from cell
            for val in range(1, row + 1):
                up_gap_penalty.append(matrix[val][col] + openGap + (row-val-1)*extGap)
            index_up = np.argmax(up_gap_penalty)
            max_up_score = max(up_gap_penalty)

            # get all values directly left from cell
            for val in range(1, col + 1):
                left_gap_penalty.append(matrix[row][val] + openGap + (col-val-1)*extGap)
            index_left = np.argmax(left_gap_penalty)
            score_left = max(left_gap_penalty)
            
            # find max value
            new_value = max(
                0,
                diagonal,
                max_up_score,
                score_left
            )
            # set new value
            matrix[row][col] = new_value
            up_gap_penalty = []
            left_gap_penalty = []

            # traceback setup
            if matrix[row][col] == 0:
                traceback_matrix[row][col] = Trace.STOP
                direction_matrix[row][col] = 0
            elif matrix[row][col] == diagonal:
                traceback_matrix[row][col] = Trace.DIAGONAL
                direction_matrix[row][col] = 1
            elif matrix[row][col] == max_up_score:
                traceback_matrix[row][col] = Trace.UP
                direction_matrix[row][col] = index_up + 1
            elif matrix[row][col] == score_left:
                traceback_matrix[row][col] = Trace.LEFT
                direction_matrix[row][col] = index_left + 1
            # identify the max score
            if matrix[row][col] >= max_score:
                max_score_indicies = (row, col)
                max_score = matrix[row][col]

    # call traceback using traceback matrix
    seq1, seq2 = traceback(traceback_matrix, max_score_indicies, sequence1, sequence2)
    return formatAlignment(sequence1, sequence2, seq1, seq2, max_score_indicies)

"""Follows traceback part of SW algorithm"""
def traceback(tracing_matrix, max_score_indicies, sequence1, sequence2):

    # Initializing variables
    final_sequence1 = ""
    final_sequence2 = ""
    next_char_1 = ""
    next_char_2 = ""
    i_start, j_start = max_score_indicies
    
    # Doing traceback
    while tracing_matrix[i_start][j_start] != 0:
        if tracing_matrix[i_start][j_start] == Trace.DIAGONAL:
            next_char_1 = sequence1[i_start - 1]
            next_char_2 = sequence2[j_start - 1]
            i_start -= 1
            j_start -= 1

        elif tracing_matrix[i_start][j_start] == Trace.UP:
            next_char_1 = sequence1[i_start - 1]
            next_char_2 = '-'
            i_start -= 1
        elif tracing_matrix[i_start][j_start] == Trace.LEFT:
            next_char_1 = '-'
            next_char_2 = sequence2[j_start-1]
            j_start -= 1
        
        final_sequence1 += next_char_1
        final_sequence2 += next_char_2

    # reverse the sequences to get the printable version
    final_sequence1 = final_sequence1[::-1]
    final_sequence2 = final_sequence2[::-1]

    return final_sequence1, final_sequence2

def formatAlignment(sequence1, sequence2, seq1_align, seq2_align, max_score_indicies):
    # Initialize variables
    seq1_length = len(sequence1) + 1
    seq2_length = len(sequence2) + 1
    i_start, j_start = max_score_indicies

    # Identify the end of the matched alignment
    seq1_align += ')' + sequence1[i_start:seq1_length]
    seq2_align += ')' + sequence2[j_start:seq2_length]

    # Find the start of the alignment
    chars_align1 = len(seq1_align.replace("-", ""))
    chars_align2 = len(seq2_align.replace("-", ""))
    
    # Idnetify the start of the matched alignment
    seq1_align = sequence1[0:(seq1_length-chars_align1)] + "(" + seq1_align
    seq2_align = sequence2[0:(seq2_length-chars_align2)] + "(" + seq2_align

    # Add spaces before the alignment, this will depend on which alignment is larger
    if len(seq1_align) >= len(seq2_align):
        spaces_in_front = seq1_align.find('(') - seq2_align.find('(')
        spaces_in_back = len(seq1_align) - seq1_align.find(')') - 1
        seq2_align = (" " * spaces_in_front) + seq2_align + (" " * spaces_in_back)
    else:
        spaces_in_front = seq2_align.find('(') - seq1_align.find('(')
        spaces_in_back = len(seq2_align) - seq2_align.find(')') - 1
        seq1_align = (" " * spaces_in_front) + seq1_align + (" " * spaces_in_back)
    
    # Add the vertical bars for matches
    match_string = ""
    if len(seq1_align) == len(seq2_align):
        for i in range(len(seq1_align)):
            if (seq1_align[i] == seq2_align[i] and seq1_align[i].isalpha()): #ensures only letters get vertical bar
                match_string += "|"
            else:
                match_string += " "

    return seq1_align, match_string, seq2_align

### Implement your Smith-Waterman Algorithm
def runSW(inputFile, scoreFile, openGap, extGap):
    with open("output.txt", "w") as f:
        sequences = printSequences(inputFile, f)
        sequence1 = sequences[0]
        sequence2 = sequences[1]

        # initializing scoring matrix
        rows = len(sequence1)
        cols = len(sequence2)

        # creating score matrix with column of 0s
        matrix = np.zeros((rows + 1, cols + 1), dtype=int)
        similarity_matrix = createSimilarityMatrix(scoreFile)
        seq1_align, match_string, seq2_align = scoreSW(openGap, extGap, sequence1, sequence2, similarity_matrix, matrix)

        ### write output
        printScoreMatrix(matrix.T, sequence1, sequence2, f)
        printBestLocalAlignment(seq1_align, match_string, seq2_align, np.amax(matrix), f)

### Run your Smith-Waterman Algorithm
runSW(args.input, args.score, args.opengap, args.extgap)

## Usage: python hw1.py -i <input file> -s <score file>
## Example: python hw1.py -i input.txt -s blosum62.txt
