import matplotlib.pyplot as plt


def plot_multivariate_timeseries(data, labels=None, title="Multivariate Time Series", figsize=(12, 6), linestyle='-', alpha=0.9):
    """
    绘制多变量时间序列图（支持缩放后数据）
    
    参数：
    - data: np.ndarray, shape=(n_samples, n_features)
    - labels: list of str，图例标签
    - title: str，图表标题
    - figsize: tuple，图表尺寸
    - linestyle: str，线型（默认实线）
    - alpha: float，透明度
    """
    n_samples, n_features = data.shape
    if labels is None:
        labels = [f"Feature {i}" for i in range(n_features)]

    assert len(labels) == n_features, "标签数必须等于特征数"

    plt.figure(figsize=figsize)
    for i in range(n_features):
        plt.plot(data[:, i], label=labels[i], linestyle=linestyle, alpha=alpha)

    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Scaled Value", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
