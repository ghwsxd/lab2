import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

# Загрузка данных
data_train = pd.read_csv('D:/hernya/adult_train.csv', sep=';')
data_test = pd.read_csv('D:/hernya/adult_test.csv', sep=';')

# Удаление строк с некорректными метками в тестовой выборке
data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target'] == ' <=50K.')]

# Перекодировка целевой переменной в числовой формат
data_train.loc[data_train['Target'] == ' <=50K', 'Target'] = 0
data_train.loc[data_train['Target'] == ' >50K', 'Target'] = 1

data_test.loc[data_test['Target'] == ' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target'] == ' >50K.', 'Target'] = 1

# Приведение типов
for col in ['Age', 'fnlwgt', 'Education_Num', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week']:
    data_test[col] = data_test[col].astype(int)

# Определение категориальных и числовых признаков
categorical_columns_train = [c for c in data_train.columns if data_train[c].dtype == 'object']
numerical_columns_train = [c for c in data_train.columns if data_train[c].dtype != 'object']

categorical_columns_test = [c for c in data_test.columns if data_test[c].dtype == 'object']
numerical_columns_test = [c for c in data_test.columns if data_test[c].dtype != 'object']

# Заполнение пропусков
for c in categorical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].mode()[0]).infer_objects(copy=False)
for c in categorical_columns_test:
    data_test[c] = data_test[c].fillna(data_test[c].mode()[0]).infer_objects(copy=False)
for c in numerical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].median())
for c in numerical_columns_test:
    data_test[c] = data_test[c].fillna(data_train[c].median())  # используем статистику train

# One-hot-кодирование категориальных признаков
data_train = pd.concat([
    data_train,
    pd.get_dummies(data_train['Workclass'], prefix="Workclass"),
    pd.get_dummies(data_train['Education'], prefix="Education"),
    pd.get_dummies(data_train['Martial_Status'], prefix="Martial_Status"),
    pd.get_dummies(data_train['Occupation'], prefix="Occupation"),
    pd.get_dummies(data_train['Relationship'], prefix="Relationship"),
    pd.get_dummies(data_train['Race'], prefix="Race"),
    pd.get_dummies(data_train['Sex'], prefix="Sex"),
    pd.get_dummies(data_train['Country'], prefix="Country")
], axis=1)

data_test = pd.concat([
    data_test,
    pd.get_dummies(data_test['Workclass'], prefix="Workclass"),
    pd.get_dummies(data_test['Education'], prefix="Education"),
    pd.get_dummies(data_test['Martial_Status'], prefix="Martial_Status"),
    pd.get_dummies(data_test['Occupation'], prefix="Occupation"),
    pd.get_dummies(data_test['Relationship'], prefix="Relationship"),
    pd.get_dummies(data_test['Race'], prefix="Race"),
    pd.get_dummies(data_test['Sex'], prefix="Sex"),
    pd.get_dummies(data_test['Country'], prefix="Country")
], axis=1)

# Удаление исходных категориальных признаков
drop_cols = ['Workclass', 'Education', 'Martial_Status', 'Occupation',
             'Relationship', 'Race', 'Sex', 'Country']
data_train.drop(drop_cols, axis=1, inplace=True)
data_test.drop(drop_cols, axis=1, inplace=True)

# Синхронизация столбцов train и test
missing_cols = set(data_train.columns) - set(data_test.columns)
for col in missing_cols:
    data_test[col] = 0
data_test = data_test[data_train.columns]

# Разделение признаков и целевой переменной
X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target'].astype(int)

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target'].astype(int)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_predictions))

# GridSearchCV для дерева решений
tree_params = {"max_depth": range(2, 11)}
locally_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17), tree_params, cv=5)
locally_best_tree.fit(X_train, y_train)
print("Best Tree Params:", locally_best_tree.best_params_)
print("Best Tree CV Score:", locally_best_tree.best_score_)

# Настроенное дерево
tuned_tree = DecisionTreeClassifier(max_depth=locally_best_tree.best_params_['max_depth'], random_state=17)
tuned_tree.fit(X_train, y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
print("Tuned Tree Accuracy:", accuracy_score(y_test, tuned_tree_predictions))

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=17)
rf.fit(X_train, y_train)
cv_scores = cross_val_score(rf, X_train, y_train, cv=3)
print("Random Forest CV Scores:", cv_scores)
print("Random Forest CV Mean:", cv_scores.mean())

forest_predictions = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, forest_predictions))

# GridSearch для случайного леса
forest_params = {"max_depth": range(10, 16), "max_features": range(5, 105, 20)}
locally_best_forest = GridSearchCV(
    RandomForestClassifier(n_estimators=10, random_state=17, n_jobs=-1),
    forest_params,
    cv=3,
    verbose=1,
)
locally_best_forest.fit(X_train, y_train)
print("Best Forest Params:", locally_best_forest.best_params_)
print("Best Forest CV Score:", locally_best_forest.best_score_)

# Настроенный случайный лес
tuned_forest_predictions = locally_best_forest.predict(X_test)
print("Tuned Forest Accuracy:", accuracy_score(y_test, tuned_forest_predictions))
