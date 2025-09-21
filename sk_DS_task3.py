#Decision Tree Classifier on Bank Marketing Dataset
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("bank.csv", sep=";")   
# Show first few rows
print("Dataset Preview:")
print(df.head())


df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("y_yes", axis=1)  
y = df_encoded["y_yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(18,10))
plot_tree(clf, feature_names=X.columns, class_names=["No","Yes"], filled=True, fontsize=8)
plt.title("Decision Tree - Bank Marketing Dataset")
plt.show()


importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)[:10]

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Top 10 Feature Importances")
plt.show()
