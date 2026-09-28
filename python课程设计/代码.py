import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import missingno as msno
import numpy as np
from scipy import stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier  # ada分类器 、 极度随机数分类算法
from sklearn.ensemble import GradientBoostingClassifier # 集成学习梯度提升决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,recall_score,precision_score,f1_score # 评估函数
from sklearn.model_selection import train_test_split  # 数据集划分函数
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier  # KNN算法
from sklearn.preprocessing import LabelEncoder  # 编码转换
from sklearn.preprocessing import StandardScaler # 数据标准化函数
from sklearn.svm import SVC  # 支持向量机
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  #二次判别分析
from xgboost import XGBClassifier  # XGB分类器
from catboost import CatBoostClassifier #CatBoost分类器
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier  #BP神经网络
from sklearn import metrics # 量度函数
from pyswarm import pso     # 粒子群算法
from sklearn.ensemble import VotingClassifier  #投票分类器
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False




df = pd.read_csv("C:\\Users\\86198\\Desktop\\shuju.csv")
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 1000)
print('数据集的形状为：\n',df.shape)
print('数据集的前10行为：\n{}'.format(df.head(10)))
print(df.describe()) 
null1 = pd.isnull(df).sum()
print('各列的缺失值数为：\n' ,null1)
churn = df['Churn'].value_counts()       
print('客户未流失数与流失数分别为\n',churn)
###查看数据类型
dfinfo = ('数据集的各列的数据类型分别为\n',df.info())


df['TotalCharges']=df['TotalCharges'].apply(pd.to_numeric,errors = 'coerce')
print('转换后的数据类型为：',df["TotalCharges"].dtypes)
print('TotalCharges列转换后的数据缺失值总计为:',pd.isnull(df["TotalCharges"]).sum())
df.dropna(inplace=True) 
print("新的数据形状为：",df.shape)


df['Churn'].replace(to_replace = 'Yes', value = 1,inplace = True)
df['Churn'].replace(to_replace = 'No', value = 0,inplace =True)
print('将Churn归一化后的Churn列为：\n',df['Churn'].head(10))




##客户流失饼状图
plt.pie(df["Churn"].value_counts(), labels=df["Churn"].value_counts().index, explode=(0.1, 0), autopct='%1.2f%%', shadow=True, colors = ['#c2c2f0','#ffb3e6', '#66b3ff'])	
plt.title("客户流失比例")	
plt.show()

##分析性别与流失的关系
plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
plt.title('性别的流失分布: Male(M), Female(F)', fontsize=15, y=1.1)
 
plt.axis('equal')
plt.tight_layout()
plt.show()

##分析用户合同签订方式与流失的关系
fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>用户合同签订方式流失分布<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

##分析用户支付方式与流失的关系
labels = df['PaymentMethod'].unique()
values = df['PaymentMethod'].value_counts()
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>用户支付方式流失分布</b>")
fig.show()
fig = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>客户支付方式与流失的分布</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


##分析客户是否有家属与流失的关系
color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>家属与流失的分布</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

##分析用户配偶与流失的关系
color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="Partner", barmode="group", title="<b>配偶与流失的分布</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()




##分析客户年龄与流失的关系
color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>年龄与流失的分布</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

##分析用户在线安全服务与流失的关系
color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group", title="<b>在线安全服务与流失的分布</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


##分析客户电子账单与流失的关系
color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="PaperlessBilling",  title="<b>电子账单与流失的分布</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()
##
##分析客户技术支持与流失的关系

fig = px.histogram(df, x="Churn", color="TechSupport",barmode="group",  title="<b>技术支持与流失的关系</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

##分析客户电话服务与流失的关系

color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="PhoneService", title="<b>电话服务与客户流失的关系</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

##分析客户停留时长与流失的关系
fig = px.box(df, x='Churn', y = 'tenure')
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
fig.update_xaxes(title_text='Churn', row=1, col=1)
fig.update_layout(autosize=True, width=750, height=600,title_font=dict(size=25, family='Courier'),title='<b>客户停留时长与流失分布</b>',)
fig.show()


##可视化"tenure"与"MonthlyCharges"和"TotalCharges"核密度图
def kdeplot(col):
    plt.figure(figsize=(20, 12), dpi=100)
    plt.title('KDE for {}'.format(col))
    ax1 = sns.kdeplot(df[df['Churn'] == 1][col], shade=True)
    ax2 = sns.kdeplot(df[df['Churn'] == 0][col], shade=True)
    plt.legend(['流失', '未流失'])
    plt.show()
 
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')

##实际每月费用与预期之差和密度图
df['total_charges_to_tenure_ratio'] = df['TotalCharges'] / df['tenure']
df['monthly_charges_diff'] = df['MonthlyCharges'] - df['total_charges_to_tenure_ratio']
kdeplot('monthly_charges_diff')


##可视化"MonthlyCharges"和"TotalCharges"与"tenure"之间的关系
graphic = sns.PairGrid(df, y_vars=["tenure"], x_vars=["MonthlyCharges", "TotalCharges"], height=4.5, hue="Churn", aspect=1.1, palette="pastel")
ax = graphic.map(plt.scatter, alpha=0.6)
plt.show()

##PhoneService、MultipleLines、InternetService与MonthlyCharges的情况
ax1 = sns.catplot(x="MultipleLines", y="MonthlyCharges", hue="Churn", kind="violin", split=True, palette="pastel", data=df, height=4.2, aspect=1.4)
ax1.set(title='MultipleLines vs MonthlyCharges')
ax2 = sns.catplot(x="InternetService", y="MonthlyCharges", hue="Churn", kind="violin", split=True, palette="pastel", data=df, height=4.2, aspect=1.4)
ax2.set(title='InternetService vs MonthlyCharges')
ax3 = sns.catplot(x="PhoneService", y="MonthlyCharges", hue="Churn", kind="violin", split=True, palette="pastel", data=df, height=4.2, aspect=1.4)
ax3.set(title='PhoneService vs MonthlyCharges')
plt.show()




##构造相关性矩阵
charges = df.iloc[:, 1:20]
corrDf = charges.apply(lambda x: pd.factorize(x)[0])


#######皮尔逊相关系数（连续型变量）
corr = corrDf.corr()
#皮尔逊相关系数的显著性检验
corr_array1 = corrDf.corr().values
p_values = np.zeros_like(corr_array1)
for i in range(corr_array1.shape[0]):
    for j in range(i+1, corr_array1.shape[1]):
        corr = corr_array1[i, j]

        df = len(corr_array1) - 2  
        t = corr * np.sqrt(df / (1 - corr**2))  
        p_value = 2 * (1 - stats.t.cdf(abs(t), df))  

        p_values[i, j] = p_value
        p_values[j, i] = p_value
print("p-value:", p_values)



#########斯皮尔曼相关系数（离散型和连续型）
corr, p_values = stats.spearmanr(corrDf)
corrDf = pd.DataFrame(corr)
##斯皮尔曼相关系数的显著性检验
corr_array2 = corrDf.corr().values
p_values = np.zeros_like(corr_array2)
for i in range(corr_array2.shape[0]):
    for j in range(i+1, corr_array2.shape[1]):
        corr = corr_array2[i, j]
        df = len(corr_array2) - 2  
        t = corr * np.sqrt(df / (1 - corr**2))  
        p_value = 2 * (1 - stats.t.cdf(abs(t), df))  
        p_values[i, j] = p_value
        p_values[j, i] = p_value
print("p-value:", p_values)



#########克莱姆相关系数
corrDf = charges.apply(lambda x: pd.factorize(x)[0])
num_vars = corrDf.shape[1]
corrMatrix = np.zeros((num_vars, num_vars))
for i in range(num_vars):
    for j in range(i, num_vars):
        if i==j:
            corrMatrix[i, j] = 1
            continue
        contingency_table = pd.crosstab(corrDf.iloc[:,i], corrDf.iloc[:,j])
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
        corrMatrix[i, j] = v
        corrMatrix[j, i] = v


# 将克莱姆相关系数矩阵转换为数据框
corrDf = pd.DataFrame(corrMatrix, columns=corrDf.columns, index=corrDf.columns)
corr_array = corrDf.corr().values
#显著性检验
p_values = np.zeros_like(corr_array)
for i in range(corr_array.shape[0]):
    for j in range(i+1, corr_array.shape[1]):
        corr = corr_array[i, j]
        df = len(corr_array) - 2  
        t = corr * np.sqrt(df / (1 - corr**2))  
        p_value = 2 * (1 - stats.t.cdf(abs(t), df))  
        p_values[i, j] = p_value
        p_values[j, i] = p_value
print("p-value:", p_values)


corrMatrix[4,:] = corr_array1[4,:]
corrMatrix[:,4] = corr_array1[:,4]
corrMatrix[-2:,:] = corr_array1[-2:,:]
corrMatrix[:,-2:] = corr_array1[:,-2:]
corrDf = pd.DataFrame(corrMatrix, columns=corrDf.columns, index=corrDf.columns)
plt.figure(figsize=(20,16))        
ax = sns.heatmap(corrDf,xticklabels = corrDf.columns,
                 yticklabels = corrDf.columns,linewidths = 0.2,
                 annot = True) 
plt.title("Correlation between variables")  
plt.show()


df = pd.read_csv("C:\\Users\\86198\\Desktop\\shuju.csv")
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 1000)
null1 = pd.isnull(df).sum()
churn = df['Churn'].value_counts()       
df['TotalCharges']=df['TotalCharges'].apply(pd.to_numeric,errors = 'coerce')
df.dropna(inplace=True) 
df['Churn'].replace(to_replace = 'Yes', value = 1,inplace = True)
df['Churn'].replace(to_replace = 'No', value = 0,inplace =True)



##one-hot编码
df_dummies = pd.get_dummies(df.iloc[:, 1:21])
df_dummies.head()


##电信用户是否流失与各变量之间的相关性
plt.figure(figsize=(15, 8))
df_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.title("电信用户是否流失与各变量之间的相关性")
plt.show()


##在线备份业务、设备保护业务、网络电视、网络电影和无互联网服务对客户流失率的影响
covariables = ["OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
covariables_dict = [ "在线备份业务", "设备保护业务", "网络电视", "网络电影"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i, item in enumerate(covariables):
    plt.subplot(2, 2, (i + 1))
    ax = sns.countplot(x=item, hue="Churn", data=df, palette="pastel", order=["Yes", "No", "No internet service"])
    plt.xlabel(covariables_dict[i])
    plt.title(covariables_dict[i] + "对客户流失率的影响")
    i = i + 1
plt.show()


######数据预处理

##删除customerID、gender、phone service
df_var = df.iloc[:, 2:20]
df_var.drop("PhoneService", axis=1, inplace=True)
##提取ID
df_id = df['customerID']

##连续型数据标准化
scaler = StandardScaler(copy=False)
scaler.fit_transform(df_var[['tenure', 'MonthlyCharges', 'TotalCharges']])
df_var[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_var[['tenure', 'MonthlyCharges', 'TotalCharges']])


##画箱型图，查看异常值
plt.figure(figsize=(8, 4))
numbox = sns.boxplot(data=df_var[['tenure', 'MonthlyCharges', 'TotalCharges']], palette="pastel")
plt.title("客户职位、月费用和总费用信息箱型图")
plt.show()


##绘制连续型数据属性的频率直方图
def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)
    plt.show()
num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')), columns=num_cols)
for feat in ['tenure', 'MonthlyCharges', 'TotalCharges']: distplot(feat, df_std, color='c')



##特征转换
##查看对象类型字段中存在的值
def uni(columnlabel):
    print(columnlabel, "--", df_var[columnlabel].unique())  # unique函数去除其中重复的元素，返回唯一值
df_object = df_var.select_dtypes(['object'])
for i in range(0, len(df_object.columns)):
    uni(df_object.columns[i])
df_var.replace(to_replace='No internet service', value='No', inplace=True)
df_var.replace(to_replace='No phone service', value='No', inplace=True)
for i in range(0, len(df_object.columns)):
    uni(df_object.columns[i])
def labelencode(columnlabel):
    df_var[columnlabel] = LabelEncoder().fit_transform(df_var[columnlabel])
 
for i in range(0, len(df_object.columns)):
    labelencode(df_object.columns[i])
 
for i in range(0, len(df_object.columns)):
    uni(df_object.columns[i])


##模型设计、选择
##建立训练集和测试集
X = df_var
y = df["Churn"].values
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=0)
print("训练数据和测试数据被分成的组数：",sss.get_n_splits(X,y))

for train_index, test_index in sss.split(X, y):
    print("train:", train_index, "test:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train,y_test=y[train_index], y[test_index]

##训练模型
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
Classifiers=[["Random Forest",RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto",max_leaf_nodes = 30)],  
    ["Support Vector Machine",SVC(random_state = 1)],
    ["LogisticRegression",LogisticRegression()],
    ["KNN",KNeighborsClassifier(n_neighbors=11)],
    ["Naive Bayes",GaussianNB()],
    ["Decision Tree",DecisionTreeClassifier()],
    ["AdaBoostClassifier", AdaBoostClassifier()],
    ["GradientBoostingClassifier", GradientBoostingClassifier()],
    ["XGB", XGBClassifier()],
    ["CatBoost", CatBoostClassifier(logging_level='Silent')],
    ["Voting Classifier",VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')],
    ["Extra Trees",ExtraTreesClassifier()],
    ["Linear Discriminant Analysis",LinearDiscriminantAnalysis()],
    ["Quadratic Discriminant Analysis",QuadraticDiscriminantAnalysis()],
    ["MLPClassifier",MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)]]
results_df = pd.DataFrame(columns=['Classifier', 'Recall', 'Precision', 'F1 Score'])
for name, classifier in Classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    recall = recall_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    results_df = results_df._append({'Classifier': name, 
                                    'Recall': recall, 
                                    'Precision': precision, 
                                    'F1 Score': f1score}, ignore_index=True)
print(results_df)

# 定义BP神经网络模型的参数空间
def create_param_space():
    space = [(50, 500), # 隐藏层神经元个数
             (0.1, 1), # 学习率
             (0.1, 1)] # 正则化系数
    return space

# 定义适应度函数（损失函数），用于评估BP神经网络模型的性能
def fitness_function(params, X_train, y_train, X_val, y_val):
    hidden_layer_sizes = int(params[0])
    learning_rate_init = params[1]
    alpha = params[2]
    
    # 训练BP神经网络模型
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), 
                        learning_rate_init=learning_rate_init, 
                        alpha=alpha, 
                        max_iter=1000, 
                        random_state=0)
    clf.fit(X_train, y_train)
    
    # 在验证集上计算适应度函数值（分类准确率的倒数）
    y_pred = clf.predict(X_val)
    return 1 - accuracy_score(y_val, y_pred)

# 定义PSO算法的回调函数，用于输出每一次迭代的最优适应度函数值和参数组合
def callback_function(xk, convergence):
    print("Best fitness:", convergence[-1])
    print("Best parameters:", xk)

# 初始化PSO算法的参数，并运行PSO算法
param_space = create_param_space()
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
xopt, fopt = pso(fitness_function,
                 lb=[s[0] for s in param_space], 
                 ub=[s[1] for s in param_space], 
                 args=(X_train, y_train, X_test, y_test), 
                 swarmsize=10, 
                 omega=options['w'], 
                 phip=options['c1'], 
                 phig=options['c2'], 
                 maxiter=100)

# 构建最优的BP神经网络模型，并在测试集上评估其性能
hidden_layer_sizes = int(xopt[0])
learning_rate_init = xopt[1]
alpha = xopt[2]
clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), 
                    learning_rate_init=learning_rate_init, 
                    alpha=alpha, 
                    max_iter=1000, 
                    random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
recall = recall_score(y_test,y_pred)  
precision = precision_score(y_test,y_pred)
f1score = f1_score(y_test,y_pred) 
print('基于PSO算法优化的BP神经网络的召回率与精准率与F1分数分别为：{}，{}， {}'.format(recall,precision,f1score))


##进行预测
pred_X = df_var.tail(10)
pre_id = df_id.tail(10)
model = GaussianNB()
model.fit(X_train,y_train)
pred_y = model.predict(pred_X)
predDf = pd.DataFrame({'customerID':pre_id, 'Churn':pred_y})
print(predDf)
