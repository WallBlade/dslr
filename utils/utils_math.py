import numpy as np

def	min_val(arr: np.ndarray) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    min_val = arr[0]

    for num in arr:
        if min_val > num:
            min_val = num
    
    return min_val

def	max_val(arr: np.ndarray) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    max_val = arr[0]

    for num in arr:
        if max_val < num:
            max_val = num
    
    return max_val

def	mean_val(arr: np.ndarray) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    
    return sum(arr) / len(arr)

def std_val(arr: np.ndarray) -> float:
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    
    mean = mean_val(arr)
    dev = 0

    for val in arr:
        dev += (val - mean) ** 2
    
    return (dev / len(arr)) ** 0.5
        

def main():
    data = np.array([2, 5, 3, 7, 8, 1, 9])
    inf = min_val(data)
    sup = max_val(data)
    moy = mean_val(data)
    print(f"min = {inf}\tmax = {sup}\tmean = {moy}")
    print(f"std = {np.std(data)}")
    print(f"std_val = {std_val(data)}")
    
if __name__ == "__main__":
    main()