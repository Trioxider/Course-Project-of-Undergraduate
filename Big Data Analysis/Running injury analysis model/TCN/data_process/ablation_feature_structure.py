import numpy as np

def create_samples(day_data, input_days=28, forecast_days=7, mask_seq_feats=None, mask_weekly_feats=None):
    """
    day_data: 原始DataFrame
    mask_seq_feats: 长度为5的布尔列表，例如 [True, False, ..., False] 表示屏蔽 total km
    mask_weekly_feats: 长度为6的布尔列表，表示每周6个特征是否屏蔽
    """
    X_seq, X_weekly, y = [], [], []

    for i in range(input_days, len(day_data) - forecast_days):
        seq_window = day_data.iloc[i - input_days:i]
        forecast_window = day_data.iloc[i:i + forecast_days]

        # 时序特征（5个）
        seq_feat = seq_window[['total km', 'km Z3-4', 'strength training', 'perceived exertion', 'perceived recovery']].values
        if mask_seq_feats:
            for idx, mask in enumerate(mask_seq_feats):
                if mask:
                    seq_feat[:, idx] = 0.0
        X_seq.append(seq_feat)

        # 周特征（6×4周）
        weekly_feats = []
        for j in range(4):
            week = seq_window.iloc[j*7:(j+1)*7]
            one_week = [
                week.shape[0],  # nr. sessions
                7 - week['strength training'].sum(),
                week['total km'].sum(),
                week['km Z3-4'].sum(),
                week['perceived exertion'].mean(),
                week['perceived recovery'].min()
            ]
            if mask_weekly_feats:
                one_week = [0.0 if mask_weekly_feats[k] else val for k, val in enumerate(one_week)]
            weekly_feats.extend(one_week)
        X_weekly.append(weekly_feats)

        # 标签
        y.append(int(forecast_window['injury'].any()))

    return np.array(X_seq), np.array(X_weekly), np.array(y)
