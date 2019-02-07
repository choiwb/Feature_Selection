import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix

sonar_df = pd.read_csv("sonar.csv",sep = ',')

print(sonar_df.head(20))
print(sonar_df.shape)

print(sonar_df['Class'].value_counts())
print(sonar_df.dtypes)
print(sonar_df.describe)
print(sonar_df.info())

# histogram
sonar_df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
plt.show()

# density plot
sonar_df.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1, figsize=(12,12))
plt.show()

# box and whisker
# sonar_df.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False, fontsize=1)
# plt.show()

# One - Hot Encoding M (Mine) and R (Rock) to 0, 1
sonar_df['Class'] = sonar_df['Class'].map(lambda x : 1 if x == 'R' else 0)
# print(sonar_df['Class'].value_counts())

# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(sonar_df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
fig.set_size_inches(10,10)
plt.show()

# 0.5 < IV < 0.9 Feature selection by smbinning R Package
sonar_df = sonar_df[['V9',	'V10',	'V36',	'V13',	'V47',	'V48',	'V49',	'V46',	'V51',	'V54',	'V45', 'Class']]

# 연속형 변수 -> binning -> 범주형 변수 by smbinning in R Package
# Optimal Binning
for i in range(sonar_df.shape[0]):
    if (sonar_df.iloc[i, 0] <= 0.116):
        sonar_df.iloc[i, 0] = 0.1
    else:
        sonar_df.iloc[i, 0] = 0.2

print(sonar_df['V9'])

# evaluate algorithm
# split out validation dataset for the end

X = sonar_df.iloc[:, 0:-1]
Y = sonar_df.iloc[:,-1]

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 0 ~ 1 Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print('Standardized X_train')
print(X_train)
X_test = sc.transform(X_test)
print('Standardized X_test')
print(X_test)

model = LogisticRegression(random_state = 0)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Confusion Matrix의 4가지 결과값 도출
cm = confusion_matrix(Y_test, y_pred)

tp = cm[0,0]
tn = cm[1,1]
fp = cm[0,1]
fn = cm[1,0]

acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp+fp)
sen = tp / (tp+fn)
spec = tn / (fp + tn)

print('정확도 (Accuracy): %f , 정밀도 (Precision): %f , 민감도 (Sensitivity): %f , 특이도 (Specificity): %f' % (acc, prec, sen, spec))

binary = np.array([[tn,fp], [fn,tp]])
fig, ax = plot_confusion_matrix(conf_mat = binary, show_absolute=True, show_normed=True, colorbar=True)
plt.show()

# ROC 커브 및 AUC 도출

FPR, TPR, thresholds = roc_curve(Y_test, y_pred)

plt.figure(figsize=(10,5))  # figsize in inches
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 50%
plt.plot(FPR, TPR, lw=2, label='Logaristic Regression (AUC = %0.2f)' % auc(FPR, TPR))
plt.title('ROC curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()