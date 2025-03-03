import zipfile
import os
import shutil

with zipfile.ZipFile('src/raw_data/train_reduced.zip', 'r') as zip:
    zip.extractall('src/train_reduced/')

os.chdir('src/train_reduced')
if '.DS_Store' in os.listdir():
    os.remove('.DS_Store')

if '__MACOSX' in os.listdir():
    shutil.rmtree('__MACOSX')

if os.path.exists('Task 21'):
    os.rename('Task 21', 'train')
