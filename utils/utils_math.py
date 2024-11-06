import numpy as np
import pandas as pd
import math as mt

def	min_val(arr) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    min_val = arr[0]

    for num in arr:
        if mt.isnan(num):
            continue
        if min_val > num:
            min_val = num
    
    return min_val

def	max_val(arr) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    max_val = arr[0]

    for num in arr:
        if mt.isnan(num):
            continue
        if max_val < num:
            max_val = num
    
    return max_val

def	mean_val(arr) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")

    return sum_val(arr) / len(arr)

def sum_val(arr) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    
    ret = 0

    for num in arr:
        if mt.isnan(num):
            continue
        ret += num

    return ret

def count_val(arr) -> float:
    count = 0

    for num in arr:
        if mt.isnan(num):
            continue
        count += 1

    return count

def std_val(arr) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")

    mean = mean_val(arr)
    dev = 0

    for num in arr:
        if mt.isnan(num):
            continue
        dev += (num - mean) ** 2

    return (dev / len(arr)) ** 0.5

def percentile_val(arr, perc) -> float:
    """
    Calculates the value at the specified percentile for the given array.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    perc (float): The percentile value between 0 and 1.
    
    Returns:
    float: The value at the specified percentile.
    """
    sorted_arr = np.sort(arr)
    index = int(len(sorted_arr) * perc)
    return float(sorted_arr[index])

def	describe(data):
    df = pd.read_csv(data)

    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols]
    
    counts = []
    means = []
    stds = []
    mins = []
    q25 = []
    q50 = []
    q75 = []
    maxs = []
    
    for column in numeric_df.columns:
        counts.append(count_val(numeric_df[column]))
        means.append(mean_val(numeric_df[column]))
        stds.append(std_val(numeric_df[column]))
        mins.append(min_val(numeric_df[column]))
        q25.append(percentile_val(numeric_df[column], 0.25))
        q50.append(percentile_val(numeric_df[column], 0.50))
        q75.append(percentile_val(numeric_df[column], 0.75))
        maxs.append(max_val(numeric_df[column]))

    # print(f'count : {counts}\n')
    # print(f'mean : {means}\n')
    # print(f'std : {stds}\n')
    # print(f'min : {mins}\n')
    print(f'25% : {q25}\n')
    print(f'50% : {q50}\n')
    print(f'75% : {q75}\n')
    # print(f'max : {maxs}\n')
    

def main():
    data = pd.read_csv("../datasets/dataset_train.csv")
    df = pd.DataFrame(data)
    print(df.describe())

    print('-----------------------------------------')

    describe('../datasets/dataset_train.csv')
    
    # inf = min_val(data)
    # sup = max_val(data)
    # moy = mean_val(data)
    # print(f"min = {inf}\tmax = {sup}\tmean = {moy}")
    # print(f"std = {np.std(data)}")
    # print(f"std_val = {std_val(data)}")
    
if __name__ == "__main__":
    main()