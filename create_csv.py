import os
import numpy as np
import pandas as pd

DATASET_PATH = r"C:\Users\kumar\OneDrive\Desktop\ml-projects-new\signlanguageAI\dataset" 

all_data = []
all_labels = []

# Loop through each gesture folder
for label in os.listdir(DATASET_PATH):
    gesture_folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(gesture_folder):
        continue

    # Loop through each .txt file inside the gesture folder
    for file in os.listdir(gesture_folder):
        if file.endswith(".txt"):
            file_path = os.path.join(gesture_folder, file)
            try:
                values = np.loadtxt(file_path)
                
                # Flatten to 63 values if needed
                values = values.flatten()

                all_data.append(values)
                all_labels.append(label)
            except:
                print(f"Error reading {file_path}")

# Convert to DataFrame
df = pd.DataFrame(all_data)
df["label"] = all_labels

# Save CSV
df.to_csv("all_gestures.csv", index=False)

print("\n all_gestures.csv generated successfully!")
print("Total samples:", len(df))
print("Gestures:", df["label"].value_counts())
