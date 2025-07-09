import pandas as pd

def Load_data():
    """
    Load the data from CSV files.
    """
    # Load the day and week data
    day_data = pd.read_csv('dataset/day_approach_maskedID_timeseries.csv')
    week_data = pd.read_csv('dataset/week_approach_maskedID_timeseries.csv')
    
    return day_data, week_data


if __name__ == '__main__':
    day_data, week_data = Load_data()
    # print the first five rows of the data
    #print(day_data.head())  # data data
    #print(week_data.head()) # week data
    print((day_data['injury']).value_counts())
    print((week_data['injury']).value_counts())

