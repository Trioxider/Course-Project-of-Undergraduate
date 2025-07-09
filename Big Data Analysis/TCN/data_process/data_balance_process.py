from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
import numpy as np

def balance_sequence_data(X_seq, X_weekly, y, method='smote_enn', random_state=42):
    """
    对 (X_seq, X_weekly, y) 进行采样增强，支持 SMOTE、欠采样、SMOTE+ENN

    参数：
        X_seq: [N, T, D]，时序特征
        X_weekly: [N, F]，每周特征
        y: [N,] 标签
        method: 'smote', 'undersample', 'smote_enn'
    返回：
        X_seq_new, X_weekly_new, y_new
    """
    N, T, D = X_seq.shape
    F = X_weekly.shape[1]

    # 拼接两个特征分支作为 SMOTE 输入
    X_combined = np.concatenate([X_seq.reshape(N, -1), X_weekly], axis=1)

    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=random_state)
    else:
        raise ValueError("method 参数必须是 'smote', 'undersample' 或 'smote_enn'")

    X_resampled, y_resampled = sampler.fit_resample(X_combined, y)

    # 还原为原始结构
    X_seq_resampled = X_resampled[:, :T * D].reshape(-1, T, D)
    X_weekly_resampled = X_resampled[:, T * D:]

    # 打乱顺序
    X_seq_resampled, X_weekly_resampled, y_resampled = shuffle(
        X_seq_resampled, X_weekly_resampled, y_resampled, random_state=random_state
    )

    return X_seq_resampled, X_weekly_resampled, y_resampled
