import numpy as np
import pandas as pd
from .load_data import Load_data

def process_dumplicated_data(day_data, week_data):
    """
    Process the duplicated data in the day and week data.
    """

    # Check for duplicated data in the day and week data
    index_duplicated_day = day_data.duplicated()  # duplicated data in the day data
    #print(index_duplicated_day, '\n')
    #print('----------------')
    
    index_duplicated_week = week_data.duplicated()  # duplicated data in the week data
    #print(index_duplicated_week, '\n')
    #print('----------------')

    # Remove duplicated data
    new_day_data = day_data.drop_duplicates()   # delete the duplicated data in the day data
    new_week_data = week_data.drop_duplicates()  # delete the duplicated data in the week data

    return new_day_data, new_week_data

def process_null_data(day_data, week_data):
    """
    Process the null data in the day and week data.
    """

    # Check for null data in the day and week data
    null_count_day = day_data.isnull().sum()  # null data in the day data
    #print(null_count_day, '\n')
    #print('----------------')

    null_count_week = week_data.isnull().sum()  # null data in the week data
    #print(null_count_week, '\n')
    #print('----------------')

    # Fill null data with 0
    new_day_data = day_data.fillna(0)  # fill null data in the day data with 0
    new_week_data = week_data.fillna(0)  # fill null data in the week data with 0

    return new_day_data, new_week_data



if __name__ == '__main__':
    # Load the data
    day_data, week_data = Load_data()
    # print the first five rows of the data
    #print(day_data.head())  # data data
    #print(week_data.head()) # week data
    # Process the duplicated data
    new_day_data, new_week_data = process_dumplicated_data(day_data, week_data)
    # print the first five rows of the data
    #print(new_day_data.head())  # data data
    #print(new_week_data.head()) # week data

    # Process the null data
    new_day_data, new_week_data = process_null_data(new_day_data, new_week_data)
    # print the first five rows of the data
    #print(new_day_data.head())  # data data
    #print(new_week_data.head()) # week data

    