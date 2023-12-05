import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import csv

# 读数据
data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

# # 年龄分布直方图
# # 首先，创建一个新的 DataFrame data_train_age，该 DataFrame 仅包含年龄不为空的数据。
# # 绘制年龄分布的直方图，使用 70 个 bins（柱子）来显示数据分布。
# # x 轴表示年龄，y 轴表示人数。
# data_train_age = data_train[data_train['Age'].notnull()]
# plt.figure(figsize=(8, 3))
# data_train_age['Age'].hist(bins=70)
# plt.xlabel('Age')
# plt.ylabel('Num')
# plt.show()

# # 按年龄分组的生存率堆叠条形图
# # x 轴表示年龄，y 轴表示人数。
# # 绘制按年龄组的生存率的堆叠条形图。
# # 绘制按年龄组的生存率的堆叠条形图。
# bins = [0, 6, 12, 20, 39, 59, 100]
# group_names = ['infant', 'child', 'teen', "prime", "middle", "old"]
# data_train['categories'] = pd.cut(data_train['Age'], bins, labels=group_names)
# plt.style.use('grayscale')
# s_pclass = data_train['Survived'].groupby(data_train['categories'])
# s_pclass = s_pclass.value_counts().unstack()
# fig = s_pclass.plot(kind='bar',
#                     stacked=True,
#                     colormap='tab20c',
#                     title='mortality rate of age',
#                     fontsize=20)
# fig.axes.title.set_size(20)
# plt.show()

# # 按社会经济地位（Pclass）和性别（Sex）的堆叠条形图：
# # 绘制按社会经济地位（Pclass）的生存率和死亡率的堆叠条形图。
# # 绘制按性别（Sex）的生存率和死亡率的堆叠条形图。
# plt.style.use('grayscale')
# s_pclass = data_train['Survived'].groupby(data_train['Pclass'])
# s_pclass = s_pclass.value_counts().unstack()
# s_sex = data_train['Survived'].groupby(data_train['Sex'])
# s_sex = s_sex.value_counts().unstack()
# fig = s_sex.plot(kind='bar',
#                  stacked=True,
#                  colormap='tab20c',
#                  title=' mortality rate of sex',
#                  fontsize=20)
# plt.show()

# fig = s_pclass.plot(kind='bar',
#                     stacked=True,
#                     colormap='tab20c',
#                     title='mortality rate of pclass',
#                     fontsize=20)
# plt.show()


# # Pclass 和 Sex 对 Age 和 Survived 的小提琴图：
# # 创建一个包含两个子图的图表，分别显示社会经济地位（Pclass）和性别（Sex）对年龄（Age）和生存（Survived）的影响。
# # 绘制小提琴图，用于展示数值数据的分布。
# fig.axes.title.set_size(20)
# fig, ax = plt.subplots(1, 2, figsize=(9, 4))
# sns.violinplot(x="Pclass",
#                y="Age",
#                hue="Survived",
#                data=data_train_age,
#                split=True,
#                ax=ax[0])
# ax[0].set_title('Pclass and Age vs Survived')
# ax[0].set_yticks(range(0, 110, 10))
# sns.violinplot(x="Sex",
#                y="Age",
#                hue="Survived",
#                data=data_train_age,
#                split=True,
#                ax=ax[1])
# ax[1].set_title('Sex and Age vs Survived')
# ax[1].set_yticks(range(0, 110, 10))
# plt.show()

# # 兄弟姐妹/配偶数量（SibSp）和父母/子女数量（Parch）对生存情况的计数图
# # 绘制兄弟姐妹/配偶数量（SibSp）和父母/子女数量（Parch）对生存情况的计数图。
# sns.countplot(x='SibSp', hue='Survived', data=data_train)
# plt.show()
# sns.countplot(x='Parch', hue='Survived', data=data_train)
# plt.show()

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
# 选择用于训练的特征（predictors）
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# 初始化逻辑回归模型并进行训练：
LogRegAlg = LogisticRegression(random_state=1)  # 初始化逻辑回归
re = LogRegAlg.fit(data_train[predictors], data_train['Survived'])
# 交叉验证评估模型性能并输出平均准确率
scores = model_selection.cross_val_score(LogRegAlg,
                                         data_train[predictors],
                                         data_train['Survived'],
                                         cv=5)  # 使用sklearn库里的交叉验证函数获取预测准确率分数
print("准确率为：")
print(scores.mean())

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
# 选择测试集的特征
test_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
test_predictors = data_test[test_features]  # 构造测试集的survived列
# 使用训练好的模型进行预测
data_test['Survived'] = LogRegAlg.predict(test_predictors)
print('对测试人员的预测：')
print(data_test)

# 写入预测结果
with open('./result/predict.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['PassengerId', 'Survived'])
    for i in range(0, data_test.shape[0]):
        writer.writerow(
            [data_test['PassengerId'][i], data_test['Survived'][i]])
