import os
import tarfile
import shutil

def preprocess_data():
    file_path = "/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar"  # Update with actual dataset path

    if os.path.exists(file_path):
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path="/kaggle/working/Training Data")  # Extract to a working directory
        print(f"Extracted training files successfully!")
    else:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Define paths
    root_dir = "/kaggle/working/Training Data"  # Update if needed
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']

    # Create modality folders if they don't exist
    for modality in modalities:
        modality_folder = os.path.join(root_dir, modality)
        os.makedirs(modality_folder, exist_ok=True)

    # Loop through patient folders
    for patient_id in os.listdir(root_dir):
        patient_folder = os.path.join(root_dir, patient_id)
        
        if os.path.isdir(patient_folder):  # Ensure it's a directory
            for file in os.listdir(patient_folder):
                for modality in modalities:
                    if file.endswith(f"_{modality}.nii.gz"):
                        src_path = os.path.join(patient_folder, file)
                        dest_path = os.path.join(root_dir, modality, file)  # Move to modality folder
                        shutil.move(src_path, dest_path)

    print("✅ All Training files organized successfully!")

    # Clean up patient folders
    for patient_id in os.listdir(root_dir):
        patient_folder = os.path.join(root_dir, patient_id)
        if patient_id not in modalities:
            if '.DS_Store' not in patient_folder:
                shutil.rmtree(patient_folder)

    for folder in os.listdir('/kaggle/working/Training Data'):
        print(f"Modality folder: {folder}")

if __name__ == "__main__":
    preprocess_data()
