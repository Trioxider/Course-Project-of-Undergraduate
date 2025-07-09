import matplotlib.pyplot as plt

def plot_training_metrics(metrics_dict):
    """
    Plots training and validation metrics over epochs.

    Args:
        metrics_dict (dict): A dictionary with keys like 'train_loss', 'val_loss',
                             'train_precision', 'val_precision', etc., and values
                             are lists of metric values per epoch.
    """
    epochs = range(1, len(next(iter(metrics_dict.values()))) + 1)
    plt.figure(figsize=(8, 6))
    num_metrics = len(metrics_dict) // 2  # assuming train/val for each metric
    for i, metric in enumerate(sorted(set(k.replace('train_', '').replace('val_', '') for k in metrics_dict))):
        plt.subplot((num_metrics + 1) // 2, 2, i + 1)
        if f'train_{metric}' in metrics_dict:
            plt.plot(epochs, metrics_dict[f'train_{metric}'], label='Train')
        if f'val_{metric}' in metrics_dict:
            plt.plot(epochs, metrics_dict[f'val_{metric}'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
