# src/download_dataset.py
import kagglehub
import shutil
import os

# 1️⃣ Download LeapGestRecog dataset (Sign Language Gesture Dataset)
print("⏳ Downloading LeapGestRecog dataset...")
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
print("Download complete!")

print("Path to dataset files:", path)

# 2️⃣ Copy the entire dataset folder into your project's /data directory
dest_path = os.path.join("data", "leapgestrecog")
os.makedirs("data", exist_ok=True)
shutil.copytree(path, dest_path, dirs_exist_ok=True)
print(f"Dataset folder copied to {dest_path}")

# 3️⃣ List a few sample files to confirm
print("\n Sample files inside dataset:")
for root, dirs, files in os.walk(dest_path):
    for f in files[:5]:
        print(" -", os.path.join(root, f))
    break  # Just show first folder
