import pandas as pd
import utils_math as mt
import math
import seaborn as sns
import matplotlib.pyplot as plt

def fill_nan(arr):
    mean = mt.mean_val(arr)
    clean_arr = arr.fillna(mean)
    
    return clean_arr

def detect_outliers(df, threshold=2.5):
    outliers = (df.abs() > threshold)
    
    return outliers
    
def visualize_outliers(df):
    sns.boxplot(data=df, orient='h')
    plt.title("Boxplot to Identify Outliers")
    plt.show()

def	prepare_data():
    df = pd.read_csv('datasets/dataset_test.csv')
    df = df.dropna(axis=1, how='all')
    normalized_df = mt.normalize(df)
    
    for column in normalized_df:
        normalized_df[column] = fill_nan(normalized_df[column])
    
    outliers = detect_outliers(normalized_df)
    print(outliers.to_string())
    # visualize_outliers(normalized_df.drop(columns=['Outlier'], errors='ignore'))
    
if __name__ == "__main__":
    prepare_data()