import pandas as pd

# log_prediction.py
import pandas as pd
import os

def log_epoch_predictions(preds, labels, epoch, save_path):
    """
    保存每一轮训练的预测结果。

    参数：
    - preds: numpy array，模型的预测结果（如 shape [batch_size]）
    - labels: numpy array，真实标签（如 shape [batch_size]）
    - epoch: int，当前 epoch 编号
    - save_path: str，CSV 文件保存路径
    """
    df_epoch = pd.DataFrame({
        'epoch': [epoch + 1] * len(preds),
        'predicted': preds.flatten(),
        'label': labels.flatten(),
        'correct': (preds.flatten() == labels.flatten()).astype(int)
    })

    # 检查文件是否存在（决定是否写入表头）
    file_exists = os.path.isfile(save_path)
    df_epoch.to_csv(save_path, mode='a', index=False, header=not file_exists)


def count_predictions(csv_path):
    """
    统计预测为 positive 和 negative 的数量。

    参数：
    - csv_path: str，CSV 文件路径

    返回：
    - 一个包含统计结果的字典
    """
    df = pd.read_csv(csv_path)

    counts = df['predicted'].value_counts().sort_index()  # 按照 0, 1 排序
    result = {
        'predicted_negative (0)': counts.get(0, 0),
        'predicted_positive (1)': counts.get(1, 0),
        'total': len(df)
    }

    return result

def count_predictions_by_epoch(csv_path):
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['epoch', 'predicted']).size().unstack(fill_value=0)
    grouped.columns = ['negative (0)', 'positive (1)']
    return grouped


