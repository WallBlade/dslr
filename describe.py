from utils import utils_math as m
import pandas as pd
import sys

def	describe(dataset_file):
    """
    Generates descriptive statistics for numeric columns in a CSV file.

    This function reads a CSV file into a DataFrame and calculates the count, mean, 
    standard deviation, minimum, 25th percentile, median (50th percentile), 
    75th percentile, and maximum for each numeric column. It returns a summary 
    DataFrame with these statistics.

    Parameters:
    data (str): The file path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: A DataFrame where each row represents a statistic 
                        (count, mean, std, min, 25%, 50%, 75%, max), and each column 
                        corresponds to a numeric column in the input data.
    """
    df = pd.read_csv(dataset_file)

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
        # Remove NaN values before calculating statistics
        column_data = numeric_df[column].dropna().values
        
        counts.append(m.count_val(column_data))
        means.append(m.mean_val(column_data))
        stds.append(m.std_val(column_data))
        mins.append(m.min_val(column_data))
        q25.append(m.percentile_val(column_data, 0.25))
        q50.append(m.percentile_val(column_data, 0.50))
        q75.append(m.percentile_val(column_data, 0.75))
        maxs.append(m.max_val(column_data))
    
    # Store the result of the computed values inside a pandas DataFrame
    # for formatting purpose
    describe_df = pd.DataFrame({
        'count': counts,
        'mean': means,
        'std': stds,
        'min': mins,
        '25%': q25,
        '50%': q50,
        '75%': q75,
        'max': maxs,
    }, index=numeric_cols).T
    
    print(describe_df)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file.csv>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
 
    describe(dataset_file)