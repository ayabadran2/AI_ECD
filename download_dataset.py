import kagglehub

print("Downloading UR Fall Detection Dataset...")

dataset_path = kagglehub.dataset_download("shahliza27/ur-fall-detection-dataset")

print(f"Dataset downloaded to: {dataset_path}")
