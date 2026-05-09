'''def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
arr = [6, 5, 12, 2, 1]
print(selection_sort(arr))'''

#0,1- Knapsack Problem
def knapsack(weight, val, k):
    n = len(weight)
    dp = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(k + 1):
            if weight[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight[i - 1]] + val[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][k]
weight=[10,20,30]
val=[12,6,8]
k=15
print(knapsack(weight, val, k))