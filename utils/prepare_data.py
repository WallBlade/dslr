import pandas as pd
import utils_math as mt
import seaborn as sns
import matplotlib.pyplot as plt

def fill_nan(arr):
    '''
    Compute the mean of an array and replace its nan values with it.
    '''
    mean = mt.mean_val(arr)
    clean_arr = arr.fillna(mean)

    return clean_arr

def detect_outliers(df, threshold=2.5):
    '''
    Search for outliers in a z-score normalized DataFrame.
    '''
    outliers = (df.abs() > threshold)
    
    return outliers
    
def visualize_outliers(df):
    '''
    Visualize outliers in a dataset with a boxplot
    '''
    sns.boxplot(data=df, orient='h')
    plt.title("Boxplot to Identify Outliers")
    plt.show()

def	prepare_data():
    '''
    Prepare the dataset for model to train: normalize, treat nan values, drop
    useless features.
    '''
    df = pd.read_csv('datasets/dataset_train.csv')
    curated_df = df.drop(['Arithmancy', 'Defense Against the Dark Arts', 'Care of Magical Creatures'], axis=1)
    curated_df = mt.normalize(curated_df)

    for column in curated_df:
        curated_df[column] = fill_nan(curated_df[column])
    
    outliers = detect_outliers(curated_df)
    print(outliers.to_string())
    visualize_outliers(curated_df.drop(columns=['Outlier'], errors='ignore'))
    
    return curated_df

def main():
    prepare_data()

if __name__ == "__main__":
    main()