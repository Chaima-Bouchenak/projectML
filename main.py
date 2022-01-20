# Project STAT3
# CHAIMA BOUCHENAK
# M2 DLAD
################################# Load libraries ################################
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import *
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap

########################### Import data and labels ############################
df = pd.read_csv('./data/data.csv')
label = pd.read_csv('./data/labels.csv')

########################### Dimension of dataset ###############################
print("Dimension of dataset : ", df.shape) # (801, 20532)

########################### Class distribution #################################
print(label.groupby('Class').size().sort_values(ascending=False))
# 5 features Y
# BRCA    300
# KIRC    146
# LUAD    141
# PRAD    136
# COAD     78

################## Extract X the data and Y the features #####################
X = df.iloc[:,1:]
Y = label.iloc[:,1]

######################  Dimensionality Reduction ##############################

# PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(X)
X_pca = pd.DataFrame(data = principalComponents)
X_pca = pd.concat([X_pca.reset_index().drop(['index'],axis=1),Y.reset_index().drop(['index'],axis=1)], axis=1)

# T-SNE reduction
X_tsne = TSNE(n_components=2).fit_transform(X)
X_tsne = pd.DataFrame(data = X_tsne)
X_tsne = pd.concat([X_tsne.reset_index().drop(['index'],axis=1),Y.reset_index().drop(['index'],axis=1)], axis=1)

sns.pairplot(x_vars=0, y_vars=1, data=X_tsne, hue="Class",palette="Set2",height=7,aspect=1.2)
plt.show()

#UMAP
sns.pairplot(x_vars=0, y_vars=1, data=X_pca, hue="Class",palette="Set2",height=7,aspect=1.2)
plt.show()

X_umap = umap.UMAP().fit_transform(X)
X_umap = pd.DataFrame(data=X_umap)
X_umap = pd.concat([X_umap.reset_index().drop(['index'],axis=1),Y.reset_index().drop(['index'],axis=1)], axis=1)

sns.pairplot(x_vars=0, y_vars=1, data=X_umap, hue="Class",palette="Set2",height=7,aspect=1.2)
plt.show()


###################### Split training and testing data #########################

X = X_umap.drop(['Class'],axis=1)
Y = X_umap['Class']




X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


######################## Standardisation ######################################
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#################### Choice of the models ############################

# Creation list of different model representez by tuples
models = []
models.append(('RandomForestClassifier', RandomForestClassifier(max_depth=2, random_state=0)))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model with - cross validation
results = []
names = []
means = -1
i = 0
for test, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(test)
    if means < cv_results.mean() :
        means = cv_results.mean()
        best_model = test
        pos = i
    i=+1
    print(' \n*** %s ***: \nPrécision : %f \nVariance: %f' % (test, cv_results.mean(), cv_results.std()))


print("\n**********\nThe model with the best precision is :", best_model, "\n**********\n")

algo_chosen = models[i][1]  # the best model

################# Cross Validation with the best model   ########################

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(algo_chosen, X_train, y_train, cv=kfold, scoring='accuracy')

print("Min of cross-validations:", cv_results.min())
print("Mean of cross-validations: ",cv_results.mean())
print("Max of cross-validations:", cv_results.max())
print("SD of cross-validations:", cv_results.std())

################## Training and Making Predictions ############################

model = algo_chosen
# Train the model
model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)


########################## Evaluate predictions ################################
print("Accuracy Score with test set:\n", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))


# Learning curve


train_sizes, train_scores, test_scores = learning_curve(model, X, Y, n_jobs=-1, cv=10, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title(best_model)
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()

# box-like grid
plt.grid()

# plot the std deviation as a transparent range at each training set size
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

# plot the average training and test score lines at each training set size
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# sizes the window for readability and displays the plot
# shows error from 0 to 1.1
plt.ylim(0,1.1)
plt.show()


if __name__ == "__main__":
    main()
