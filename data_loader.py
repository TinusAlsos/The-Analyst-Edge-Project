import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API credentials
api = KaggleApi()
api.authenticate()

# Set up file paths
data_dir = 'data'
dataset_name = 'dataset-name'  # Replace with the name of the dataset you want to download
zip_file_path = os.path.join(data_dir, dataset_name + '.zip')
csv_file_path = os.path.join(data_dir, dataset_name + '.csv')

# Download dataset as a zip file
api.dataset_download_files(dataset_name, path=data_dir)

# Extract zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Rename extracted file to CSV
os.rename(os.path.join(data_dir, dataset_name + '.csv'), csv_file_path)

# Remove zip file
os.remove(zip_file_path)
