import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_absolute_error


def fft_mae(df, df_reduced):
    fft_original = np.abs(np.fft.fft(df["pressure"]))
    fft_reduced = np.abs(np.fft.fft(df_reduced["pressure"]))

    min_len = min(len(fft_original), len(fft_reduced))
    return mean_absolute_error(fft_original[:min_len], fft_reduced[:min_len])


def main():
    WORK_DIR = r"../src/raw_data/train"
    RES_DIR = r"../src/raw_data/train_reduced.zip"

    currentFile = os.listdir(WORK_DIR)[0]
    threshold = 0.01

    df = pd.read_csv(os.path.join(WORK_DIR, currentFile), sep="\\s+", header=None, names=["time", "pressure"])

    pressure_diffs = df["time"].diff().dropna()
    meanTimeDiff = pressure_diffs.mean()

    df_reduced = df.loc[(df["pressure"].diff().abs() > threshold).fillna(True)]
    error = fft_mae(df, df_reduced)

    while threshold <= meanTimeDiff and error <= 1000:

        threshold *= 1.1
        df_reduced = df.loc[(df["pressure"].diff().abs() > threshold).fillna(True)]
        error = fft_mae(df, df_reduced)

    print(error, threshold)

    with open(currentFile + "tmp", "w") as file:
        file.write(df_reduced.to_string())

    with zipfile.ZipFile(RES_DIR, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(currentFile + "tmp", currentFile)

    os.remove(currentFile + "tmp")

if __name__ == '__main__':
    main()