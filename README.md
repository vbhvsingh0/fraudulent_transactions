# fraudulent_transactions

A few models were developed based on Decision trees, Logistic Regression, and clustering models to categorize fraudulent transactions

A. Supervised machine learning models

The code inside 'supervised' named folder uses the 'fraud_sampledata.csv' to model the supervised machine learning model to categorize fraud transactions. It has used 3 models:
    
	a. Logistic regression 
	b. random Forest classifier
 	c. Ensemble model combining above two.

In terms of performance, the model 'c' was the best having 94 % recall value.

B. Unsupervised machine learning models

The script is present inside the 'unsupervised' named folder. Here, a types of clustering models were used as given below:
	
	a. Kmeans clustering method
	b. DBSCAN clustering method

In terms of performance, b worked better suggesting , the shape of the features might be convex shaped.