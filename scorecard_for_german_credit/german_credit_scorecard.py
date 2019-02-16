import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, confusion_matrix, auc

# df = pd.read_csv("original_dt_woe.csv", sep = ",")
# df = pd.read_csv("iv_dt_woe.csv", sep = ",")
# df = pd.read_csv("step_dt_woe.csv", sep = ",")
# df = pd.read_csv("smote_dt_woe.csv", sep = ",")
df = pd.read_csv("smote_iv_dt_woe.csv", sep = ",")
print(df.head())

X = df.iloc[:,0:-1]
print(X.head())
print(X.shape)
Y = df.iloc[:,-1]
print(Y.head())
print(Y.shape)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# 0 ~ 1 Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print('Standardized X_train')
print(X_train)
X_test = sc.transform(X_test)
print('Standardized X_test')
print(X_test)


model = LogisticRegression()
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