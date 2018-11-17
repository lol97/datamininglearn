import pandas as pd
import pickle as pkl

data = pd.read_csv("EEG Eye State.csv")
feature = data
feature = feature.drop(['eyeDetection'],axis=1)
label = data['eyeDetection'].values

# from sklearn import svm
# clf = svm.SVC(gamma='scale')
# clf.fit(feature, label)
# # print(clf)
# # pkl.dump(clf,open('modelSVM.sav','wb'))

# true_predict = label
# train_predict = clf.predict(feature)

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(true_predict, train_predict))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(feature, label).predict(feature)
print(y_pred)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(label, y_pred))