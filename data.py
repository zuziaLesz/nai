import kagglehub
import os
import pandas as pd

dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
train_dir = os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")

dataset_path_train = os.path.join(train_dir, "train")
dataset_path_valid = os.path.join(train_dir, "valid")

def create_dataset_csv(dataset_dir, output_csv):
    image_paths = []
    labels = []

    for subfolder in os.listdir(dataset_dir):
        subfolder_path = os.path.join(dataset_dir, subfolder)

        # Ensure it's a directory (not a file)
        if os.path.isdir(subfolder_path):
            # Define the label: 'healthy' if folder name ends with 'healthy', else 'not_healthy'
            label = 'healthy' if subfolder.endswith('healthy') else 'not_healthy'

            # Iterate through the images in each subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):  # Adjust extensions if needed
                    # Get the full image path
                    image_path = os.path.join(subfolder_path, file)

                    # Append the image path and label to the lists
                    image_paths.append(image_path)
                    labels.append(label)

    # Create a DataFrame with the image paths and labels
    data = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })

    # Save the DataFrame to a CSV file
    data.to_csv(output_csv, index=False)
    print(f"Dataset CSV has been created at {output_csv}.")


# Generate CSV files for both 'train' and 'valid' directories
create_dataset_csv(dataset_path_train, 'train_labels.csv')
create_dataset_csv(dataset_path_valid, 'valid_labels.csv')