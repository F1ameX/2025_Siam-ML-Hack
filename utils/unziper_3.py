import zipfile
import os
import shutil

def extract(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_path)

    os.chdir(extract_path)
    for item in ['.DS_Store', '__MACOSX']:
        path = os.path.join(extract_path, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

extract('src/raw_data/train_denoised.zip', 'src/train_denoised/')
extract('src/raw_data/test_denoised.zip', 'src/test_denoised/')