# Credit-Card-Fraud-Detection
This Python code uses logistic regression to build a fraud detection model on a credit card dataset. It addresses imbalanced data by creating a balanced sample and evaluates the model's accuracy on both training and testing data. The logistic regression model is trained, and its performance is assessed using accuracy metrics.

Dataset Link - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The code is Break-down Below for Understanding

1. **Data Loading and Preprocessing:**
   - The code starts by importing necessary libraries and loading a credit card dataset from a CSV file using Pandas.
   - Missing values in specific columns (`V23`, `V24`, `V25`, `V26`, `V27`, `V28`, `Amount`, and `Class`) are imputed with the mean of each respective column.
   - The 'Class' column is then converted to integers.

2. **Handling Imbalanced Data:**
   - It acknowledges that the data is imbalanced with a large number of normal transactions (Class=0) and a low number of fraud transactions (Class=1).
   - To address this, the code samples a balanced dataset by selecting a random subset of normal transactions equal to the number of fraud transactions.

3. **Creating a Balanced Dataset:**
   - A sample of normal transactions (`Legit_sample`) is obtained by randomly selecting 103 samples from the original normal transactions.
   - This sampled dataset is then concatenated with the fraud transactions to create a new dataset (`newdf`), aiming for a balanced distribution.

4. **Feature-Target Split:**
   - The dataset is split into features (`X`) and the target variable (`y`), where 'Class' is the target variable.

5. **Train-Test Split:**
   - The data is split into training and testing sets using the `train_test_split` function with a test size of 20%, stratified sampling, and a random seed for reproducibility.

6. **Logistic Regression Model Initialization:**
   - A logistic regression model is initialized with a maximum number of iterations (`max_iter`) set to 1000 to address convergence warnings.

7. **Model Training:**
   - The logistic regression model is trained using the training data (`X_train` and `y_train`) with the `fit` method.

8. **Model Evaluation on Training Data:**
   - The model's performance is evaluated on the training data by calculating the accuracy score using the `accuracy_score` function.

9. **Model Evaluation on Testing Data:**
   - Similarly, the model's performance is evaluated on the testing data, and the accuracy score is calculated.

10. **Print Results:**
    - The training and testing accuracy scores are printed to the console for analysis.

