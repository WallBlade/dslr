import pandas as pd
import utils.utils_math as mt
import seaborn as sns
import matplotlib.pyplot as plt

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
                df.loc[mask, column] = house_means[house]

def replace_outliers(arr, tolerance=2):
    # Check if arr is empty
    if len(arr) == 0:
        return arr

    mean = mt.mean_val(arr)

    # Create a mask for values outside the tolerance range
    mask = (arr >= tolerance) | (arr <= -tolerance)
    arr_processed = arr.copy()

    # Replace outliers with lower or upper bounds
    arr_processed[mask & (arr >= tolerance)] = mean
    arr_processed[mask & (arr <= -tolerance)] = mean
    
    return arr_processed

def detect_outliers(df):
    '''
    Search for outliers in a z-score normalized DataFrame.
    '''

    df_cleaned = df.copy()

    for column in df.columns[1:]:

        for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
            # Select house-specific column data
            house_mask = df['Hogwarts House'] == house
            house_column = df_cleaned.loc[house_mask, column]
            
            # Replace outliers for this house's column
            df_cleaned.loc[house_mask, column] = replace_outliers(house_column)

    return df_cleaned
    
def prepare_data(path):
    '''
    Prepare the dataset for model to train: normalize, treat nan values, drop
    useless features.
    '''
    df = pd.read_csv(path)
    
    house_column = df['Hogwarts House']
    curated_df = df.drop(['Arithmancy', 'Defense Against the Dark Arts', 'Care of Magical Creatures'], axis=1)
    
    numeric_cols = curated_df.select_dtypes(include=['number']).drop('Index', axis=1)
    
    curated_df = mt.normalize(numeric_cols)
    curated_df.insert(0, 'Hogwarts House', house_column, allow_duplicates=False)

    fill_nan(curated_df)
    curated_df = detect_outliers(curated_df)
    
    return curated_df