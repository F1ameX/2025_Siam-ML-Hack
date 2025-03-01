import zipfile
import os
import shutil

os.chdir('../')
with zipfile.ZipFile('src/raw_data/task_21.zip', 'r') as zip_ref:
    zip_ref.extractall('src/raw_data/')

os.chdir('src/raw_data')
if '.DS_Store' in os.listdir():
    os.remove('.DS_Store')

if '__MACOSX' in os.listdir():
    shutil.rmtree('__MACOSX')

if os.path.exists('Task 21'):
    os.rename('Task 21', 'train')
