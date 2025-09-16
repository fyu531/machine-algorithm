import math
import random
from collections import defaultdict

def train_test_split(data, test_size=0.3, random_state=None):
    """
    将数据集分割为训练集和测试集
    
    参数:
        data: 数据集，包含'features'和'labels'键
        test_size: 测试集比例
        random_state: 随机种子，用于重现结果
    
    返回:
        训练集和测试集，每个都是包含'features'和'labels'的字典
    """
    if random_state is not None:
        random.seed(random_state)
    
    # 生成索引并打乱
    indices = list(range(len(data['features'])))
    random.shuffle(indices)
    
    # 分割索引
    split_index = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    # 创建训练集和测试集
    train_features = [data['features'][i] for i in train_indices]
    train_labels = [data['labels'][i] for i in train_indices]
    test_features = [data['features'][i] for i in test_indices]
    test_labels = [data['labels'][i] for i in test_indices]
    
    return {
        'train': {'features': train_features, 'labels': train_labels},
        'test': {'features': test_features, 'labels': test_labels}
    }

def accuracy_score(y_true, y_pred):
    """计算准确率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_score(y_true, y_pred, average='macro'):
    """计算精确率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    classes = list(set(y_true))
    precision = {}
    
    for cls in classes:
        # 真正例
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        # 假正例
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        
        if tp + fp == 0:
            precision[cls] = 0.0
        else:
            precision[cls] = tp / (tp + fp)
    
    if average == 'macro':
        return sum(precision.values()) / len(precision)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fp_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average is None:
        return precision
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def recall_score(y_true, y_pred, average='macro'):
    """计算召回率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    classes = list(set(y_true))
    recall = {}
    
    for cls in classes:
        # 真正例
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        # 假负例
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        if tp + fn == 0:
            recall[cls] = 0.0
        else:
            recall[cls] = tp / (tp + fn)
    
    if average == 'macro':
        return sum(recall.values()) / len(recall)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fn_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average is None:
        return recall
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def f1_score(y_true, y_pred, average='macro'):
    """计算F1分数"""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    f1 = {}
    for cls in precision:
        if precision[cls] + recall[cls] == 0:
            f1[cls] = 0.0
        else:
            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
    
    if average == 'macro':
        return sum(f1.values()) / len(f1)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fp_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        fn_total = fp_total  # 对于多类分类这并不准确，但为了简化实现
        if tp_total + (fp_total + fn_total) / 2 == 0:
            return 0.0
        return 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    elif average is None:
        return f1
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值长度必须相同")
    
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

def normalize_features(features):
    """标准化特征，使每个特征的均值为0，标准差为1"""
    if not features or not features[0]:
        return features
    
    num_samples = len(features)
    num_features = len(features[0])
    
    # 计算每个特征的均值和标准差
    means = []
    stds = []
    
    for i in range(num_features):
        feature_values = [features[j][i] for j in range(num_samples)]
        mean = sum(feature_values) / num_samples
        std = math.sqrt(sum((x - mean) **2 for x in feature_values) / num_samples)
        
        # 避免除以零
        if std < 1e-10:
            std = 1.0
        
        means.append(mean)
        stds.append(std)
    
    # 标准化特征
    normalized = []
    for sample in features:
        normalized_sample = [(sample[i] - means[i]) / stds[i] for i in range(num_features)]
        normalized.append(normalized_sample)
    
    return normalized

def euclidean_distance(x1, x2):
    """计算欧氏距离"""
    if len(x1) != len(x2):
        raise ValueError("两个向量的维度必须相同")
    
    return math.sqrt(sum((a - b)** 2 for a, b in zip(x1, x2)))

def manhattan_distance(x1, x2):
    """计算曼哈顿距离"""
    if len(x1) != len(x2):
        raise ValueError("两个向量的维度必须相同")
    
    return sum(abs(a - b) for a, b in zip(x1, x2))

def majority_vote(labels):
    """多数投票决定最终标签"""
    vote_counts = defaultdict(int)
    for label in labels:
        vote_counts[label] += 1
    
    max_count = max(vote_counts.values())
    winners = [label for label, count in vote_counts.items() if count == max_count]
    
    # 如果有多个获胜者，随机选择一个
    return random.choice(winners)

def entropy(labels):
    """计算熵"""
    if not labels:
        return 0.0
    
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    entropy = 0.0
    total = len(labels)
    
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def gini_impurity(labels):
    """计算基尼不纯度"""
    if not labels:
        return 0.0
    
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    impurity = 1.0
    total = len(labels)
    
    for count in label_counts.values():
        p = count / total
        impurity -= p **2
    
    return impurity

def split_dataset(features, labels, feature_idx, threshold):
    """根据特征和阈值分割数据集"""
    left_features, left_labels = [], []
    right_features, right_labels = [], []
    
    for feature, label in zip(features, labels):
        if feature[feature_idx] < threshold:
            left_features.append(feature)
            left_labels.append(label)
        else:
            right_features.append(feature)
            right_labels.append(label)
    
    return left_features, left_labels, right_features, right_labels

def one_hot_encode(labels):
    """对标签进行独热编码"""
    unique_labels = list(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    encoded = []
    for label in labels:
        vector = [0] * len(unique_labels)
        vector[label_to_index[label]] = 1
        encoded.append(vector)
    
    return encoded, unique_labels

def one_hot_decode(encoded, unique_labels):
    """对独热编码进行解码"""
    decoded = []
    for vector in encoded:
        max_index = vector.index(max(vector))
        decoded.append(unique_labels[max_index])
    return decoded

def standard_scaler(features):
    """标准化特征（使均值为0，标准差为1）"""
    if not features or len(features[0]) == 0:
        return features
    
    n_samples = len(features)
    n_features = len(features[0])
    
    # 计算每个特征的均值和标准差
    means = [0.0] * n_features
    stds = [1.0] * n_features
    
    for i in range(n_features):
        # 计算均值
        mean = sum(features[j][i] for j in range(n_samples)) / n_samples
        means[i] = mean
        
        # 计算标准差
        squared_diff = sum((features[j][i] - mean) **2 for j in range(n_samples))
        std = math.sqrt(squared_diff / n_samples)
        if std > 1e-10:  # 避免除以零
            stds[i] = std
    
    # 标准化
    scaled_features = []
    for sample in features:
        scaled_sample = [(sample[i] - means[i]) / stds[i] for i in range(n_features)]
        scaled_features.append(scaled_sample)
    
    return scaled_features, means, stds

def min_max_scaler(features, feature_range=(0, 1)):
    """最小-最大缩放（使特征值落在指定范围内）"""
    if not features or len(features[0]) == 0:
        return features
    
    min_val, max_val = feature_range
    n_samples = len(features)
    n_features = len(features[0])
    
    # 计算每个特征的最小值和最大值
    mins = [float('inf')] * n_features
    maxs = [float('-inf')] * n_features
    
    for i in range(n_features):
        for j in range(n_samples):
            if features[j][i] < mins[i]:
                mins[i] = features[j][i]
            if features[j][i] > maxs[i]:
                maxs[i] = features[j][i]
    
    # 缩放
    scaled_features = []
    for sample in features:
        scaled_sample = []
        for i in range(n_features):
            if maxs[i] - mins[i] < 1e-10:  # 避免除以零
                scaled = (min_val + max_val) / 2
            else:
                scaled = min_val + (sample[i] - mins[i]) * (max_val - min_val) / (maxs[i] - mins[i])
            scaled_sample.append(scaled)
        scaled_features.append(scaled_sample)
    
    return scaled_features, mins, maxs

def cross_validation_split(data, n_folds=5, random_state=None):
    """将数据集分割为交叉验证的折"""
    if random_state is not None:
        random.seed(random_state)
    
    # 生成索引并打乱
    indices = list(range(len(data['features'])))
    random.shuffle(indices)
    
    # 计算每个折的大小
    fold_size = len(indices) // n_folds
    folds = []
    
    for i in range(n_folds):
        # 计算测试集索引
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(indices)
        test_indices = indices[start:end]
        
        # 训练集索引是所有不在测试集的索引
        train_indices = [idx for idx in indices if idx not in test_indices]
        
        # 创建训练集和测试集
        train_features = [data['features'][idx] for idx in train_indices]
        train_labels = [data['labels'][idx] for idx in train_indices]
        test_features = [data['features'][idx] for idx in test_indices]
        test_labels = [data['labels'][idx] for idx in test_indices]
        
        folds.append({
            'train': {'features': train_features, 'labels': train_labels},
            'test': {'features': test_features, 'labels': test_labels}
        })
    
    return folds
