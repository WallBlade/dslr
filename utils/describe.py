import utils_math as um
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
    df = df.dropna(axis=1, how='all')

    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols]

    counts = []
    means = []
    stds = []
    ranges = []
    variances = []
    null_percentages = []
    mins = []
    q25 = []
    q50 = []
    q75 = []
    maxs = []
    
    for column in numeric_df.columns:
        # Remove NaN values before calculating statistics
        column_data = numeric_df[column].dropna().values
        if len(column_data) == 0:
            continue
        
				# Caculate the number of nan values in column
        count_nan = numeric_df[column].isna().sum()

        counts.append(um.count_val(column_data))
        means.append(um.mean_val(column_data))
        stds.append(um.std_val(column_data))
        ranges.append(um.max_val(column_data) - um.min_val(column_data))
        variances.append(um.std_val(column_data) ** 2)
        null_percentages.append(100 * count_nan / len(numeric_df[column]))
        mins.append(um.min_val(column_data))
        q25.append(um.percentile_val(column_data, 0.25))
        q50.append(um.percentile_val(column_data, 0.50))
        q75.append(um.percentile_val(column_data, 0.75))
        maxs.append(um.max_val(column_data))
    
    # Store the result of the computed values inside a pandas DataFrame
    # for formatting purpose
    describe_df = pd.DataFrame({
        'count': counts,
        'mean': means,
        'std': stds,
        'range': ranges,
        'variance': variances,
        'null %': null_percentages,
        'min': mins,
        '25%': q25,
        '50%': q50,
        '75%': q75,
        'max': maxs,
    }, index=numeric_cols).T

    print(describe_df.to_string())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file.csv>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
 
    describe(dataset_file)