import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Impute missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Convert Class column to integer
df['Class'] = df['Class'].astype(int)

# Split the data into legitimate and fraud transactions
LegitData = df[df.Class == 0]
FraudData = df[df.Class == 1]

# Sample the legitimate data to balance the dataset
Legit_sample = LegitData.sample(n=len(FraudData))

# Concatenate the sampled legitimate data with the fraud data
newdf = pd.concat([Legit_sample, FraudData], axis=0)

# Split data into features and target
X = newdf.drop(columns='Class', axis=1)
y = newdf['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train the GradientBoostingClassifier model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Initialize SVM model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Create an ensemble of models
ensemble_model = VotingClassifier(estimators=[('gb', gb_model), ('svm', svm_model)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Model evaluation for training data
ensemble_train_predictions = ensemble_model.predict(X_train)
ensemble_train_accuracy = accuracy_score(ensemble_train_predictions, y_train)
print("Ensemble Training Accuracy:", ensemble_train_accuracy)

# Model evaluation for testing data
ensemble_test_predictions = ensemble_model.predict(X_test)
ensemble_test_accuracy = accuracy_score(ensemble_test_predictions, y_test)
print("Ensemble Testing Accuracy:", ensemble_test_accuracy)
