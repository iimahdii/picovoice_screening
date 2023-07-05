from typing import List
import numpy as np

def prob_rain_more_than_n(p: List[float], n: int) -> float:
    days = len(p)  # number of days in a year
    # ------------------------------
    # Law of Total Probability : 
    # Let A1, A2, ..., An be mutually exclusive and exhaustive events (i.e., they cover all possible outcomes), 
    # and let B be an event of interest. The Law of Total Probability states that:
    # P(B) = P(B | A1) * P(A1) + P(B | A2) * P(A2) + ... + P(B | An) * P(An)
    # ------------------------------
    # Initialize a two-dimensional dynamic programming table.
    # dp[i][j] represents the probability that it rains exactly j days in the first i days.
    dp = np.zeros((days+1, days+1))

    # Base case: the probability that it rains 0 days in the first 0 days is 1.
    dp[0][0] = 1.0

    # Dynamic programming.
    for i in range(1, days+1):
        # Probability it doesn't rain on day i
        dp[i][0] = dp[i-1][0] * (1 - p[i-1])

    for i in range(1, days+1):
        for j in range(1, i+1):
            # The probability that it rains j days in the first i days can be decomposed into two cases:
            # 1. It rains on day i, and it rains j-1 days in the first i-1 days.
            # 2. It doesn't rain on day i, and it rains j days in the first i-1 days.
            dp[i][j] = p[i-1] * dp[i-1][j-1] + (1 - p[i-1]) * dp[i-1][j]

    # The probability that it rains more than n days is the sum of the probabilities that it rains exactly j days for all j > n.
    prob = np.sum(dp[days][n+1:])

    return prob
