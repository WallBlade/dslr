import numpy as np
import pandas as pd
import math as mt

def	min_val(arr) -> float:
    """
    Computes the lowest value contained inside an array.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The lowest value inside the given array.
    """
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
    """
    Computes the highest value contained inside an array.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The highest value inside the given array.
    """
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
    """
    Computes the sum of all values inside the array, skipping empty rows.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The sum of all numerical values contained inside the given array.
    """
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    ret = 0

    for num in arr:
        # Only add if `num` is a real number
        if pd.notna(num):
            ret += num
    return ret

def count_val(arr) -> float:
    """
    Counts how many values there are in the array, skipping the nan.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The number of non empty values inside the array.
    """
    count = 0

    for num in arr:
        if mt.isnan(num):
            continue
        count += 1

    return count

def std_val(arr) -> float:
    """
    Calculates the value of the standard deviation for the given array.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The value of the computed standard deviation.
    """
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

def z_score(col, mean, std):
    return[((x - mean) / std) if pd.notna(x) else None for x in col]

def normalize(dataset):
    normalized_df = dataset.copy()
    
    for column in dataset.columns:
        column_data = dataset[column]
        mean = mean_val(column_data)
        std = std_val(column_data)
        
        normalized_df[column] = (column_data - mean) / std
    
    return normalized_df