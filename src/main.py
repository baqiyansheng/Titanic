import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import csv
data_train=pd.read_csv("./train.csv")
data_test=pd.read_csv("./test.csv")
 
 
 
data_train_age = data_train[data_train['Age'].notnull()]
plt.figure(figsize=(8,3))
data_train_age['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')
plt.show()
bins = [0,6, 12, 20,39,59,100]
group_names = ['infant', 'child', 'teen',"prime","middle","old"]
data_train['categories'] = pd.cut(data_train['Age'], bins, labels = group_names)
plt.style.use('grayscale')
s_pclass= data_train['Survived'].groupby(data_train['categories'])
s_pclass = s_pclass.value_counts().unstack()
fig = s_pclass.plot(kind='bar',stacked = True, colormap='tab20c',title='mortality rate of age',fontsize=20)
fig.axes.title.set_size(20)
plt.show()
plt.style.use('grayscale')
s_pclass= data_train['Survived'].groupby(data_train['Pclass'])
s_pclass = s_pclass.value_counts().unstack()
s_sex = data_train['Survived'].groupby(data_train['Sex'])
s_sex = s_sex.value_counts().unstack()
fig = s_sex.plot(kind='bar',stacked = True, colormap='tab20c',title=' mortality rate of sex',fontsize=20)
plt.show()
fig = s_pclass.plot(kind='bar',stacked = True, colormap='tab20c',title='mortality rate of pclass',fontsize=20)
plt.show()
fig.axes.title.set_size(20)
fig,ax = plt.subplots(1,2, figsize = (9,4))
sns.violinplot(x="Pclass",y="Age",hue="Survived",data=data_train_age,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot(x="Sex",y="Age",hue="Survived",data=data_train_age,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
sns.countplot(x='SibSp',hue='Survived',data=data_train)
plt.show()
sns.countplot(x='Parch',hue='Survived',data=data_train)
plt.show()
 
data_train['Age']=data_train['Age'].fillna(data_train['Age'].mean())#用年龄的平均数填充年龄空值
data_train['Embarked']=data_train['Embarked'].fillna('S')#Embarked缺失值用最多的‘S’进行填充
 
data_train.describe()
data_train.loc[data_train['Sex']=='male','Sex']=0
data_train.loc[data_train['Sex']=='female','Sex']=1
#Embarked处理：用0，1，2
data_train.loc[data_train['Embarked']=='S','Embarked']=0
data_train.loc[data_train['Embarked']=='C','Embarked']=1
data_train.loc[data_train['Embarked']=='Q','Embarked']=2
 
	
predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
LogRegAlg=LogisticRegression(random_state=1)#初始化逻辑回归
re=LogRegAlg.fit(data_train[predictors],data_train['Survived'])
 
scores=model_selection.cross_val_score(LogRegAlg,data_train[predictors],data_train['Survived'],cv=5)#使用sklearn库里的交叉验证函数获取预测准确率分数
print("准确率为：")
print(scores.mean())
 
data_test.describe()
data_test['Age']=data_test['Age'].fillna(data_test['Age'].median())
data_test['Fare']=data_test['Fare'].fillna(data_test['Fare'].max())
data_test.loc[data_test['Sex']=='male','Sex']=0
data_test.loc[data_test['Sex']=='female','Sex']=1
data_test['Embarked']=data_test['Embarked'].fillna('S')
data_test.loc[data_test['Embarked']=='S','Embarked']=0
data_test.loc[data_test['Embarked']=='C','Embarked']=1
data_test.loc[data_test['Embarked']=='Q','Embarked']=2
test_features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
test_predictors=data_test[test_features]#构造测试集的survived列
data_test['Survived']=LogRegAlg.predict(test_predictors)
print('对测试人员的预测：')
print(data_test)

with open('./result/predict.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['PassengerId','Survived'])
    for i in range(0,data_test.shape[0]):
        writer.writerow([data_test['PassengerId'][i],data_test['Survived'][i]])