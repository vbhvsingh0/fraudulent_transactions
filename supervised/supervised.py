mport pandas as pd
df = pd.read_csv("assets/creditcard_sampledata.csv")

# Explore the features available in your dataframe
print(df.info())

# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
print(occ)

# Print the ratio of fraud cases
print(occ /len(df))

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import Pipeline

# Dividing the training and test data

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Define the resampling method
method = SMOTE()

# Create the resampled feature set
X_resampled, y_resampled = method.fit_resample(X, y)

# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Obtain model predictions
predicted = model.predict(X_test)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

# Note: the above training can be combined in 1 code as below
# Define which resampling method and which ML model to use in the pipeline
#resampling = SMOTE()          # remove # to work
#model = LogisticRegression()  # remove # to work

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
#pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])   # remove # to work

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
#pipeline.fit(X_train, y_train)        # remove # to work
#predicted = pipeline.predict(X_test)  # remove # to work

# Obtain the results from the classification report and confusion matrix 
#print('Classifcation report:\n', classification_report(y_test, predicted))  # remove # to work
#conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted) # remove # to work
#print('Confusion matrix:\n', conf_mat) # remove # to work

# Using random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Define the model as the random forest
model = RandomForestClassifier(random_state=5)

# Fit the model to our training set
model.fit(X_train, y_train)

# Obtain predictions from the test data 
predicted = model.predict(X_test)



# Obtain the predictions from our random forest model 
predicted = model.predict(X_test)

# Predict probabilities
probs = model.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs[:,1]))
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))
# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, predicted)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)

# Rebalance the data weight and run again

# Change the model options
model2 = RandomForestClassifier(bootstrap=True, class_weight={0:1, 1:12}, criterion='entropy',
			
			# Change depth of model
            max_depth=10,
		
			# Change the number of samples in leaf nodes
            min_samples_leaf=10, 

			# Change the number of trees to use
            n_estimators=20, n_jobs=-1, random_state=5)

# Fit the model and get_model_results

model2.fit(X_train, y_train)

# Obtain the predictions from our random forest model 
predicted2 = model.predict(X_test)

# Predict probabilities
probs2 = model.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs2[:,1]))
print(classification_report(y_test, predicted2))
print(confusion_matrix(y_test, predicted2))
# Calculate average precision and the PR curve
average_precision2 = average_precision_score(y_test, predicted)

# Obtain precision and recall 
precision2, recall2, _ = precision_recall_curve(y_test, predicted)

# use gridsearchcv to tunr the hyperparameters

# Define the parameter sets to test
param_grid = {'n_estimators': [1, 30], 'max_features': ['auto', 'log2'],  'max_depth': [4, 8], 'criterion': ['gini', 'entropy']
}

# Define the model to use
model = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)
CV_model.best_params_

# Input the optimal parameters in the model
model3 = RandomForestClassifier(class_weight={0:1,1:12}, criterion='gini',
            n_estimators=30, max_features='log2',  min_samples_leaf=10, max_depth=8, n_jobs=-1, random_state=5)

model3.fit(X_train, y_train)

# Obtain the predictions from our random forest model 
predicted3 = model3.predict(X_test)
probs3 = model3.predict_proba(X_test)
# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs3[:,1]))
print(classification_report(y_test, predicted3))
print(confusion_matrix(y_test, predicted3))

# Ensemble of different methods

from sklearn.ensemble import VotingClassifier

# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1, 1:15}, random_state=5)
clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, criterion='gini', max_depth=8, max_features='log2',
            min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)
clf3 = DecisionTreeClassifier(random_state=5, class_weight="balanced")

# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 4, 1], flatten_transform=True)

ensemble_model.fit(X_train, y_train)

# Obtain the predictions from our random forest model 
predicted4 = ensemble_model.predict(X_test)
probs4 = ensemble_model.predict_proba(X_test)
# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs4[:,1]))
print(classification_report(y_test, predicted4))
print(confusion_matrix(y_test, predicted4))