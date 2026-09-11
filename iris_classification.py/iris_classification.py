import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())


# 2. Dataset information
print("\nDataset Information:")
print(df.info())


# 3. Statistical information
print("\nStatistical Information:")
print(df.describe())


# 4. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 5. Check species
print("\nsspecies:")
print(df["species"].value_counts())

# 6. Visualization
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
plt.show()


# 7. Select features and target
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["species"]


# 8. Convert species names into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# 9. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 10. Create model
model = RandomForestClassifier(random_state=42)


# 11. Train model
model.fit(X_train, y_train)


# 12. Make predictions
y_pred = model.predict(X_test)


# 13. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# 14. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)


# 15. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 16. Predict a new flower
new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

prediction = model.predict(new_flower)

predicted_species = encoder.inverse_transform(prediction)

print("\nNew Flower Measurements:")
print("Sepal Length: 5.1")
print("Sepal Width: 3.5")
print("Petal Length: 1.4")
print("Petal Width: 0.2")

print("\nPredicted Species:", predicted_species[0])