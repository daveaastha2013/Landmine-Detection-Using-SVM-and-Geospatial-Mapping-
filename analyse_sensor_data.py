import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import folium

# Generating random sensor data (can be replaced with actual sensor data)
np.random.seed(42)
data = {
    "latitude": np.random.uniform(10.0, 50.0, 1000),
    "longitude": np.random.uniform(10.0, 50.0, 1000),
    "metal_detection_intensity": np.random.uniform(0, 1, 1000),
    "ground_density": np.random.uniform(0, 1, 1000),
    "is_landmine": np.random.choice([0, 1], size=1000, p=[0.9, 0.1]),
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Adding Labels and features
X = df[["latitude", "longitude", "metal_detection_intensity", "ground_density"]]
y = df["is_landmine"]

# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training an SVM classifier
clf = SVC(kernel="rbf", probability=True, random_state=42)
clf.fit(X_train, y_train)

# Making predictions
y_pred = clf.predict(X_test)

# Finding accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Filtering data only for "is landmine"
landmine_data = df[df["is_landmine"] == 1]

# Plotting detected landmines on a geographical map (using Folium)
map_center = [df["latitude"].mean(), df["longitude"].mean()]
landmine_map = folium.Map(location=map_center, zoom_start=6)

# Adding markers for detected landmines
for _, row in landmine_data.iterrows():
    folium.Marker(location=[row["latitude"], row["longitude"]], 
                  popup="Landmine Detected", 
                  icon=folium.Icon(color="red")).add_to(landmine_map)

# Saving the map as HTML file
landmine_map.save("landmine_map.html")
print("Landmine map saved as 'landmine_map.html'")

# Example usage: Predict on new data
new_data = np.array([[25.0, 35.0, 0.8, 0.6]])
prediction = clf.predict(new_data)
prediction_proba = clf.predict_proba(new_data)
print("Prediction (1 = Landmine, 0 = Safe):", prediction[0])
print("Prediction Probability:", prediction_proba)
