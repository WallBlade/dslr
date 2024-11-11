import pandas as pd
import matplotlib.pyplot as plt
from utils import utils_math as mt
import mplcursors as mpl

def z_score_normalize(column, mean, std):
    # Apply z-score normalization
    return [(x - mean) / std if pd.notna(x) else None for x in column]

def	get_score(df):
    course_divergence = {}
    for course in df.columns[1:]:
        # Calculate the sum of scores for each house
        house_sums = {
            'Gryffindor': mt.sum_val(df[course][df['Hogwarts House'] == 'Gryffindor']),
            'Hufflepuff': mt.sum_val(df[course][df['Hogwarts House'] == 'Hufflepuff']),
            'Ravenclaw': mt.sum_val(df[course][df['Hogwarts House'] == 'Ravenclaw']),
            'Slytherin': mt.sum_val(df[course][df['Hogwarts House'] == 'Slytherin'])
        }

        # Calculate the mean of all four houses sums
        course_mean = mt.mean_val(list(house_sums.values()))

        # Calculate the divergence score for [course]
        divergence = 0
        for house, score in house_sums.items():
            divergence += (score - course_mean) ** 2
        divergence /= 4

        course_divergence[course] = {
            'mean': course_mean,
            'divergence': divergence
        }
    
    return pd.DataFrame(course_divergence)

def clean_data(data):
    numeric_cols = data.select_dtypes(include=['number']).columns.drop(['Index'])
    normalized_data = pd.DataFrame()
    for column in numeric_cols:
        # print(column)
        column_data = data[column]
        mean = mt.mean_val(column_data)
        std = mt.std_val(column_data)
        
        # Normalize the column
        normalized_column = z_score_normalize(column_data, mean, std)
        
        # Add normalized column to the DataFrame
        normalized_data[column] = normalized_column
      
    house_col = data['Hogwarts House']
    sanitized = pd.concat([house_col, normalized_data], axis=1)

    sanitized.to_csv('san.csv')

    return sanitized

def draw_histogram(divergence):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract course titles (keys of the divergence dictionary)
    courses = divergence.columns
    divergences = divergence.iloc[1]

    # Plot the histogram
    bars = ax.bar(courses, divergences)

    
    # Add labels and title
    ax.set_xlabel("Courses")
    ax.set_ylabel("Divergence")
    ax.set_title("Divergence Score per Course")
    
    # Rotate x-axis labels if there are many courses for better readability
    plt.xticks(rotation=45, ha="right")

    # Add hover functionality
    cursor = mpl.cursor(bars, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set(
        text=f"Divergence: {divergences.iloc[sel.index]:.2f}",
        bbox=dict(facecolor="white", edgecolor="black")
    ))

    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('datasets/dataset_train.csv')
    df = clean_data(data)
    divergence = get_score(df)
    draw_histogram(divergence)