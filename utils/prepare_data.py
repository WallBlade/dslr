import pandas as pd
import utils_math as mt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fill_nan(df):
    for column in df.columns[1:]:
        house_means = {
            'Gryffindor': mt.mean_val(df[column][df['Hogwarts House'] == 'Gryffindor']),
            'Hufflepuff': mt.mean_val(df[column][df['Hogwarts House'] == 'Hufflepuff']),
            'Ravenclaw': mt.mean_val(df[column][df['Hogwarts House'] == 'Ravenclaw']),
            'Slytherin': mt.mean_val(df[column][df['Hogwarts House'] == 'Slytherin'])
        }

        for house in house_means:
                mask = (df['Hogwarts House'] == house) & (df[column].isna())
                print(mask.to_string())
                df.loc[mask, column] = house_means[house]

def detect_outliers(df):
    '''
    Search for outliers in a z-score normalized DataFrame.
    '''

    lower_bound = []
    upper_bound = []

    for column in df:
        lower_bound.append(mt.percentile_val(df[column], 0.02))
        upper_bound.append(mt.percentile_val(df[column], 0.98))

    lower_bound = pd.Series(lower_bound, index=df.columns)
    upper_bound = pd.Series(upper_bound, index=df.columns)

    for column in df.columns:
        df[column] = df[column].apply(
            lambda x: lower_bound[column] if x < lower_bound[column] else
                      (upper_bound[column] if x > upper_bound[column] else x)
        )

    return df
    
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
    
    house_column = df['Hogwarts House']
    curated_df = df.drop(['Arithmancy', 'Defense Against the Dark Arts', 'Care of Magical Creatures'], axis=1)
    
    numeric_cols = curated_df.select_dtypes(include=['number']).drop('Index', axis=1)
    
    curated_df = mt.normalize(numeric_cols)
    curated_df.insert(0, 'Hogwarts House', house_column, allow_duplicates=False)

    fill_nan(curated_df)

    # curated_df = detect_outliers(curated_df)
    # visualize_outliers(curated_df.drop(columns=['Outlier'], errors='ignore'))
    # print(curated_df.to_string())
    return curated_df

def main():
    prepare_data()

if __name__ == "__main__":
    main()