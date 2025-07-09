import numpy as np

def create_samples(day_data, input_days=28, forecast_days=7):
    X_seq = []      # TCN input
    X_weekly = []   # FC input
    y = []          # labels

    for i in range(input_days, len(day_data) - forecast_days):
        seq_window = day_data.iloc[i - input_days:i]
        forecast_window = day_data.iloc[i:i + forecast_days]

        # day features (5)
        seq_feat = seq_window[['total km', 'km Z3-4', 'strength training', 'perceived exertion', 'perceived recovery']].values
        X_seq.append(seq_feat)

        # weekly features (6)
        weekly_feats = []
        for j in range(4):
            week = seq_window.iloc[j*7:(j+1)*7]
            weekly_feats.extend([
                week.shape[0],  # nr. sessions
                7 - week['strength training'].sum(),  #  strength_training=0 remaining days
                week['total km'].sum(),
                week['km Z3-4'].sum(),
                week['perceived exertion'].mean(),
                week['perceived recovery'].min()
            ])
        X_weekly.append(weekly_feats)

        # labels
        y.append(int(forecast_window['injury'].any()))

    return np.array(X_seq), np.array(X_weekly), np.array(y)
