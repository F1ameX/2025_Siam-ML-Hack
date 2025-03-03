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

    files = os.listdir(WORK_DIR)

    for i in range(0, len(files)):

        currentFile = files[i]

        df = pd.read_csv(os.path.join(WORK_DIR, currentFile), sep="\\s+", header=None, names=["time", "pressure"])

        pressure_diffs = df["pressure"].diff().abs().dropna()
        threshold = min(pressure_diffs[pressure_diffs > 0])

        meanDiff = pressure_diffs.mean()

        df_reduced = df.loc[(df["pressure"].diff().abs() > threshold).fillna(True)]
        error = fft_mae(df, df_reduced)

        while threshold < meanDiff and error < 1000 and threshold != 0:

            threshold *= 1.1
            df_reduced = df.loc[(df["pressure"].diff().abs() > threshold).fillna(True)]
            error = fft_mae(df, df_reduced)

        print(currentFile, error, threshold)

        with open(currentFile + "tmp", "w") as file:
            file.write(df_reduced.to_string(index = False, header = False))

        with zipfile.ZipFile(RES_DIR, "a", compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(currentFile + "tmp", currentFile)

        os.remove(currentFile + "tmp")


if __name__ == '__main__':
    main()