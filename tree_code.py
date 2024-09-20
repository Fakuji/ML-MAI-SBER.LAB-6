import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    # Сначала проверим, есть ли вообще вариативность в признаках
    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None
    
    # Сортируем вектора признаков совместно с целевым вектором
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    # Рассчитываем средние значения между соседними значениями признака
    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    
    # Гини для левого и правого сплита, умноженные на вес левого и правого
    gini_left = np.cumsum(sorted_targets)[:-1] / np.arange(1, len(sorted_targets))
    gini_left = 1 - gini_left**2 - (1 - gini_left)**2
    
    gini_right = (np.cumsum(sorted_targets[::-1])[:-1])[::-1] / np.arange(len(sorted_targets) - 1, 0, -1)
    gini_right = 1 - gini_right**2 - (1 - gini_right)**2
    
    # Рассчитываем взвешенный Gini как средневзвешенное обоих сплитов
    n_left = np.arange(1, len(sorted_targets))
    n_right = len(sorted_targets) - n_left
    weighted_ginis = (n_left*gini_left + n_right*gini_right) / len(sorted_targets)
    
    # Ищем наилучшее значение индекса (самая низкая оценка Гини)
    best_idx = np.argmin(weighted_ginis)
    
    # Возвращаем пороги, значения Гини, лучший порог и оценку Гини для лучшего порога
    return thresholds, weighted_ginis, thresholds[best_idx], weighted_ginis[best_idx]


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.

        """
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {
                    key: clicks.get(key, 0) / count for key, count in counts.items()
                }
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini < gini_best: # 1
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [
                        k for k, v in categories_map.items() if v < threshold
                    ]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node): 
    # Если узел терминальный - возвращаем класс
      if node['type'] == 'terminal':
        return node['class']
    
    # Если узел не терминальный, опускаемся по дереву
      if self._feature_types[node['feature_split']] == 'real':
        # Проверяем, меньше ли значение признака порогового значения
        if x[node['feature_split']] < node['threshold']:
            return self._predict_node(x, node['left_child'])
        else:
            return self._predict_node(x, node['right_child'])
      else:
        # Проверяем, равно ли значение признака одному из категорий, ведущих налево
        if x[node['feature_split']] in node['categories_split']:
            return self._predict_node(x, node['left_child'])
        else:
            return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
