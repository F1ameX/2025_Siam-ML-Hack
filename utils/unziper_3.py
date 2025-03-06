import zipfile
import os
import shutil

def extract(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_path)

    os.chdir(extract_path)
    if '.DS_Store' in os.listdir():
        os.remove('.DS_Store')

    if '__MACOSX' in os.listdir():
        shutil.rmtree('__MACOSX')

extract('src/raw_data/train_denoised.zip', 'src/train_denoised/')
extract('src/raw_data/test_denoised.zip', 'src/test_denoised/')