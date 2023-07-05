# Probability of Rain on More than N Days in a Year Q1
This repository contains a Python function calculates the probability that it will rain on more than n days in a given year in Vancouver.

The function takes as input an array of probabilities **p** (length 365), where each entry **p[i]** is the probability of rain on the **i-th** day of the year, and an integer **n**, the number of days. It returns the probability that it will rain on more than n days in the year.

## Algorithm Explanation
The function employs a dynamic programming approach to build a 2D probability matrix. This matrix, **dp**, is constructed such that **dp[i][j]** represents the probability that it will rain exactly **j** days in the first **i** days of the year.

To build this matrix, we iterate through each day and each possible number of rainy days up to that point. The probability that it will rain exactly **j** days in the first **i** days can be decomposed into two scenarios:

1. It rains on the **i-th** day and it rains exactly **j - 1** days in the first **i - 1** days.
2. It does not rain on the **i-th** day and it rains exactly **j** days in the first **i - 1** days.

We calculate the probabilities for each of these scenarios and sum them to find **dp[i][j]**.

Once the dp matrix is built, we find the probability that it will rain on more than **n** days by summing the probabilities **dp[365][j]** for all **j > n**.
# Word Combination Finder Q2
This project contains a Python solution to the problem of finding all possible combinations of words from a pronunciation dictionary that can produce a specific phoneme sequence.

## Solution
The solution involves creating a dictionary from the pronunciation dictionary, where the key is the sequence of phonemes for each word, and the value is the list of words that correspond to this sequence.

The function **find_word_combos_with_pronunciation** takes a sequence of phonemes as input and uses this dictionary to find all possible combinations of words that produce the input phoneme sequence. This is done by recursive backtracking.

The solution also includes a function for preprocessing the pronunciation dictionary into the required format.
# Connectionist Temporal Classification (CTC) Q4
This code contains an implementation of the (CTC) model with BLSTM networks, as described in Graves, A. et al., 2006. This model is used for speech recognition tasks, specifically for phonetic labelling on the TIMIT speech corpus.

## Overview
CTC is a powerful algorithm for sequence prediction problems because it allows for efficient calculation of gradients of the cost with respect to the model parameters and makes no assumptions about the alignments between the inputs and target sequences. This makes it ideal for speech recognition, where the alignment between the audio and transcription is unknown. The model used in this repository includes bidirectional LSTM layers, which enable the network to have access to both past and future context.
