import math
import random
from collections import defaultdict
from backend.utils import entropy, gini_impurity, split_dataset, majority_vote

class DecisionTreeNode:
    """决策树节点类"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        # 用于分裂的特征索引
        self.feature_idx = feature_idx
        # 分裂阈值
        self.threshold = threshold
        # 左子树
        self.left = left
        # 右子树
        self.right = right
        # 叶节点的值（类别）
        self.value = value

class DecisionTree:
    """决策树分类器"""
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        # 树的最大深度
        self.max_depth = max_depth
        # 分裂所需的最小样本数
        self.min_samples_split = min_samples_split
        # 不纯度计算标准 ('gini' 或 'entropy')
        self.criterion = criterion
        # 决策树的根节点
        self.root = None
    
    def _calculate_impurity(self, labels):
        """计算不纯度"""
        if self.criterion == 'gini':
            return gini_impurity(labels)
        elif self.criterion == 'entropy':
            return entropy(labels)
        else:
            raise ValueError(f"不支持的不纯度计算标准: {self.criterion}")
    
    def _information_gain(self, parent_labels, left_labels, right_labels):
        """计算信息增益"""
        parent_impurity = self._calculate_impurity(parent_labels)
        n = len(parent_labels)
        n_left, n_right = len(left_labels), len(right_labels)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        return parent_impurity - (n_left / n) * self._calculate_impurity(left_labels) - \
               (n_right / n) * self._calculate_impurity(right_labels)
    
    def _find_best_split(self, features, labels):
        """找到最佳分裂点"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = len(features), len(features[0]) if features else 0
        
        # 遍历每个特征
        for feature_idx in range(n_features):
            # 获取该特征的所有值
            feature_values = [features[i][feature_idx] for i in range(n_samples)]
            # 获取唯一值作为可能的阈值
            thresholds = list(set(feature_values))
            
            # 遍历每个可能的阈值
            for threshold in thresholds:
                # 分裂数据集
                left_features, left_labels, right_features, right_labels = \
                    split_dataset(features, labels, feature_idx, threshold)
                
                # 计算信息增益
                gain = self._information_gain(labels, left_labels, right_labels)
                
                # 更新最佳分裂点
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, features, labels, depth=0):
        """递归构建决策树"""
        n_samples = len(features)
        n_classes = len(set(labels))
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # 创建叶节点，值为多数类
            leaf_value = majority_vote(labels)
            return DecisionTreeNode(value=leaf_value)
        
        # 找到最佳分裂点
        best_feature_idx, best_threshold, best_gain = self._find_best_split(features, labels)
        
        # 如果没有信息增益，创建叶节点
        if best_gain <= 0:
            leaf_value = majority_vote(labels)
            return DecisionTreeNode(value=leaf_value)
        
        # 分裂数据集
        left_features, left_labels, right_features, right_labels = \
            split_dataset(features, labels, best_feature_idx, best_threshold)
        
        # 递归构建左右子树
        left_subtree = self._build_tree(left_features, left_labels, depth + 1)
        right_subtree = self._build_tree(right_features, right_labels, depth + 1)
        
        # 返回当前节点
        return DecisionTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, features, labels):
        """训练决策树"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        self.root = self._build_tree(features, labels)
    
    def _predict_sample(self, sample, node):
        """预测单个样本"""
        # 如果是叶节点，返回其值
        if node.value is not None:
            return node.value
        
        # 否则递归预测
        feature_value = sample[node.feature_idx]
        if feature_value < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def predict(self, features):
        """预测多个样本"""
        if self.root is None:
            raise ValueError("决策树尚未训练，请先调用fit方法")
        
        return [self._predict_sample(sample, self.root) for sample in features]
