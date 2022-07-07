import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm,neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,auc,roc_curve
import warnings


sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

data = 'features_for_detection_with_labels.csv'
df = pd.read_csv(data)

X = df.drop('L', axis=1)
y = df['L']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

print(X_train.shape)
print(X_test.shape)

smo = SMOTE(random_state=42)
X_sampling,y_sampling = smo.fit_resample(X_train,y_train)
X_train = X_sampling
y_train = y_sampling

cols = X_train.columns


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

#NBC
clf = BernoulliNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)


#SVM

clf = svm.SVC(C=1, kernel='rbf', gamma=10, decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

#KNN
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

# RF
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
y_pred = rfc_100.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

feature_scores = pd.Series(rfc_100.feature_importances_, index=X_train.columns).sort_values(ascending=False)
pd.set_option('display.max_rows', 10)
f, ax = plt.subplots(figsize=(30, 24))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=df)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()

#BP
def loaddataset(filename):
	fp = open(filename)
	dataset = []
	labelset = []
	for i in fp.readlines():
		a = i.strip().split()
		dataset.append([float(j) for j in a[:len(a)-1]])
		labelset.append(int(float(a[-1])))
	return dataset, labelset
 

def parameter_initialization(x, y, z):
	value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)
	value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)
	weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)
	weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)
	return weight1, weight2, value1, value2
 
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
 
def trainning(dataset, labelset, weight1, weight2, value1, value2):
	x = 0.1
	for i in range(len(dataset)):
		inputset = np.mat(dataset[i]).astype(np.float64)
		outputset = np.mat(labelset[i]).astype(np.float64)
		input1 = np.dot(inputset, weight1).astype(np.float64)
		output2 = sigmoid(input1 - value1).astype(np.float64)
		input2 = np.dot(output2, weight2).astype(np.float64)
		output3 = sigmoid(input2 - value2).astype(np.float64)

		a = np.multiply(output3, 1 - output3)
		g = np.multiply(a, outputset - output3)
		b = np.dot(g, np.transpose(weight2))
		c = np.multiply(output2, 1 - output2)
		e = np.multiply(b, c)
 
		value1_change = -x * e
		value2_change = -x * g
		weight1_change = x * np.dot(np.transpose(inputset), e)
		weight2_change = x * np.dot(np.transpose(output2), g)

		value1 += value1_change
		value2 += value2_change
		weight1 += weight1_change
		weight2 += weight2_change
	return weight1, weight2, value1, value2
 
def testing(dataset, labelset, weight1, weight2, value1, value2):

    rightcount = 0
    y_pred = [] #np.random.randint(2,3,dataset.shape[0])
    for i in range(len(dataset)):

        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        if output3 > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset[i] == flag:
            rightcount += 1

        print("predition is %d,fact is %d"%(flag, labelset[i]))
        y_pred.append(flag)
		

    return y_pred #rightcount / len(dataset), y_pred

if __name__ == '__main__':
	for hidden in [3,5,7,10,12,15,17,20]:
		weight1, weight2, value1, value2 = parameter_initialization(len(X_train[0]), hidden, 1)
		for i in range(1000):
			weight1, weight2, value1, value2 = trainning(X_train, y_train, weight1, weight2, value1, value2)
		y_pred = testing(X_test, y_test, weight1, weight2, value1, value2)
		print("hidden=%d\n" %(hidden))
		print(classification_report(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
 