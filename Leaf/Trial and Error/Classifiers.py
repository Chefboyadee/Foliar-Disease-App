# # Import necessary libraries
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# # Load the dataset
# df = pd.read_csv('Dataset_3.csv')
# # Display the first few rows of the dataset
# print("First few rows of the dataset:")
# print(df.head())

# # Display the columns of the dataset
# print("\nColumns of the dataset:")
# print(df.columns)

# #Remove unnecessary column
# df.drop('Unnamed: 0', axis=1, inplace=True)

# # Check for missing values
# print("\nMissing values in the dataset:")
# print(df.isnull().sum())

# # Print unique categories in the 'classlabel' column
# unique_categories = df['classlabel'].unique()
# print("Unique categories in the 'classlabel' column:")
# for category in unique_categories:
#     print(category)

# # Normalize numerical features
# scaler = StandardScaler()
# numerical_cols = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio', 'rectangularity', 'circularity', 'major_axis', 'minor_axis', 'convex_area', 'convex_ratio', 'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', 'contrast', 'correlation', 'inverse_difference_moments', 'entropy']
# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# # Splitting the data into features (X) and target variable (y)
# X = df.drop('classlabel', axis=1)
# y = df['classlabel']

# # Applying one-hot encoding to the categorical target variable
# y_encoded = pd.get_dummies(y)

# # Splitting the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Predicting the class labels for the test set
# y_pred = rf_classifier.predict(X_test)

# # Calculate metrics
# accuracy_rfc = accuracy_score(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
# precision_rfc = precision_score(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
# recall_rfc = recall_score(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
# f1_rfc = f1_score(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')

# # Print metrics
# print("Accuracy:", accuracy_rfc)
# print("Precision:", precision_rfc)
# print("Recall:", recall_rfc)
# print("F1 Score:", f1_rfc)

# # Model evaluation
# print("Classification Report:")
# print(classification_report(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), target_names=y_encoded.columns))

# # Initialize and train the K-Nearest Neighbors classifier
# knn_classifier = KNeighborsClassifier(n_neighbors=5)
# knn_classifier.fit(X_train, y_train)

# # Predicting the class labels for the test set
# y_pred_knn = knn_classifier.predict(X_test)

# # Calculate metrics for KNN
# accuracy_knn = accuracy_score(y_test.values.argmax(axis=1), y_pred_knn.argmax(axis=1))
# precision_knn = precision_score(y_test.values.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')
# recall_knn = recall_score(y_test.values.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')
# f1_knn = f1_score(y_test.values.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')

# # Print metrics for KNN
# print("K-Nearest Neighbors Metrics:")
# print("Accuracy:", accuracy_knn)
# print("Precision:", precision_knn)
# print("Recall:", recall_knn)
# print("F1 Score:", f1_knn)

# # Detailed classification report for Logistic Regression
# print("\nClassification Report for Logistic Regression:")
# print(classification_report(y_test.values.argmax(axis=1), y_pred_knn.argmax(axis=1), target_names=y_encoded.columns))


# # Initialize and train the Decision Tree classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)
# dt_classifier.fit(X_train, y_train)

# # Predicting the class labels for the test set
# y_pred_dt = dt_classifier.predict(X_test)

# # Calculate metrics for Decision Tree
# accuracy_dt = accuracy_score(y_test.values.argmax(axis=1), y_pred_dt.argmax(axis=1))
# precision_dt = precision_score(y_test.values.argmax(axis=1), y_pred_dt.argmax(axis=1), average='weighted')
# recall_dt = recall_score(y_test.values.argmax(axis=1), y_pred_dt.argmax(axis=1), average='weighted')
# f1_dt = f1_score(y_test.values.argmax(axis=1), y_pred_dt.argmax(axis=1), average='weighted')

# # Print metrics for Decision Tree
# print("Decision Tree Metrics:")
# print("Accuracy:", accuracy_dt)
# print("Precision:", precision_dt)
# print("Recall:", recall_dt)
# print("F1 Score:", f1_dt)

# # Detailed classification report for Logistic Regression
# print("\nClassification Report for Logistic Regression:")
# print(classification_report(y_test.values.argmax(axis=1), y_pred_dt.argmax(axis=1), target_names=y_encoded.columns))

# # Store the metrics in lists
# classifiers = ['Random Forest', 'KNN', 'Decision Tree']
# accuracy_scores = [accuracy_rfc, accuracy_knn, accuracy_dt]
# precision_scores = [precision_rfc, precision_knn, precision_dt]
# recall_scores = [recall_rfc, recall_knn, recall_dt]
# f1_scores = [f1_rfc, f1_knn, f1_dt]

# # Plotting the metrics
# import matplotlib.pyplot as plt

# # Bar width
# bar_width = 0.2
# index = range(len(classifiers))

# # Plotting Accuracy
# plt.bar(index, accuracy_scores, width=bar_width, label='Accuracy')
# plt.xlabel('Classifier')
# plt.ylabel('Accuracy Score')
# plt.title('Accuracy Comparison')
# plt.xticks(index, classifiers)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plotting Precision
# plt.bar(index, precision_scores, width=bar_width, label='Precision')
# plt.xlabel('Classifier')
# plt.ylabel('Precision Score')
# plt.title('Precision Comparison')
# plt.xticks(index, classifiers)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plotting Recall
# plt.bar(index, recall_scores, width=bar_width, label='Recall')
# plt.xlabel('Classifier')
# plt.ylabel('Recall Score')
# plt.title('Recall Comparison')
# plt.xticks(index, classifiers)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plotting F1 Score
# plt.bar(index, f1_scores, width=bar_width, label='F1 Score')
# plt.xlabel('Classifier')
# plt.ylabel('F1 Score')
# plt.title('F1 Score Comparison')
# plt.xticks(index, classifiers)
# plt.legend()
# plt.tight_layout()
# plt.show()
import sklearn;
print(sklearn.__version__)