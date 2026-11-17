import kagglehub

print("Downloading UR Fall Detection Dataset...")

# This will download the dataset directly into your VS Code project folder
# It automatically handles the extraction too!
dataset_path = kagglehub.dataset_download("shahliza27/ur-fall-detection-dataset")

print(f"Dataset downloaded to: {dataset_path}")
print("You can now use the .mp4 or .avi files inside this folder for testing.")