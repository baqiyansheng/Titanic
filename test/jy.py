# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv

# 加载数据
data_train = pd.read_csv('data/train.csv')  # 请将文件路径更改为实际路径
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# 处理空白值
data_train['Age'] = data_train['Age'].fillna(
    data_train['Age'].mean())  # 用年龄的平均数填充年龄空值
data_train['Embarked'] = data_train['Embarked'].fillna(
    'S')  # Embarked缺失值用频率最高的‘S’进行填充

# 数据集描述统计：
data_train.describe()
# 将性别（Sex）和上船港口（Embarked）转换为数值
data_train.loc[data_train['Sex'] == 'male', 'Sex'] = 0
data_train.loc[data_train['Sex'] == 'female', 'Sex'] = 1
# Embarked处理：用0，1，2
data_train.loc[data_train['Embarked'] == 'S', 'Embarked'] = 0
data_train.loc[data_train['Embarked'] == 'C', 'Embarked'] = 1
data_train.loc[data_train['Embarked'] == 'Q', 'Embarked'] = 2

X = data_train[features]
y = data_train[target]

# 将文本特征转换为数值（例如，使用独热编码）
X = pd.get_dummies(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# 初始化随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy}')


# 加载测试集
data_test = pd.read_csv('data/test.csv')  # 调整文件路径

# 执行与训练集相同的数据预处理步骤
# 描述统计信息
data_test.describe()
# 处理测试集中的缺失值
data_test['Age'] = data_test['Age'].fillna(
    data_test['Age'].median())  # 对测试集中的年龄（'Age'）使用中位数进行填充
data_test['Fare'] = data_test['Fare'].fillna(
    data_test['Fare'].max())  # 对票价（'Fare'）使用最大值进行填充。
data_test.loc[data_test['Sex'] == 'male',
              'Sex'] = 0  # 对上船港口（'Embarked'）使用最常见的 'S' 进行填充。
data_test.loc[data_test['Sex'] == 'female', 'Sex'] = 1
# 将性别（'Sex'）和上船港口（'Embarked'）转换为数值
data_test['Embarked'] = data_test['Embarked'].fillna('S')
data_test.loc[data_test['Embarked'] == 'S', 'Embarked'] = 0
data_test.loc[data_test['Embarked'] == 'C', 'Embarked'] = 1
data_test.loc[data_test['Embarked'] == 'Q', 'Embarked'] = 2


# 选择用于预测的特征
test_features = data_test[[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
]]

# 使用独热编码将分类特征转换为数值
test_features = pd.get_dummies(test_features, columns=['Sex', 'Embarked'])

# 确保测试集经过独热编码后具有与训练集相同的列
missing_columns = set(X_train.columns) - set(test_features.columns)
for column in missing_columns:
    test_features[column] = 0

# 使用训练好的随机森林模型进行预测
predictions = rf_model.predict(test_features)

# 创建包含 PassengerId 和预测 Survived 列的 DataFrame
result_df = pd.DataFrame({
    'PassengerId': data_test['PassengerId'],
    'Survived': predictions
})

# 将预测保存到 CSV 文件
result_df.to_csv('result/jypredictions.csv', index=False)
