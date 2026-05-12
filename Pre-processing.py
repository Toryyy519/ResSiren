import librosa
import librosa.display
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Test the property of the dataset by using single audio
file_path = "D:\\Petra\\UrbanSound8K\\UrbanSound8K\\audio\\fold1\\7383-3-0-0.wav"
y_org, sr_org = librosa.load(file_path, sr= None)
print(f"The original sampling rate is {sr_org} Hz")

# Data pre-processing: Convert to single-channel Log-Mel Spectral
def preprocess_audio(file_path, target_sr = 16000, n_mels = 128):
    try:
        # y represents Audio Time Series, sr represents Sampling Rate
        # Combine into mono "mono = True"
        y, sr = librosa.load(file_path, sr = target_sr, mono = True)
        #Construct 128 mel spectrogram
        S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = n_mels, fmax = sr//2)
        # Convert to Log-Mel
        log_S = librosa.power_to_db(S, ref = np.max)

        return log_S
    except Exception as e:
        print(f"Error processing {file_path}:{e}")
        return None

# Visualization mel spectrogram
def visualization_mel_spectrogram(log_S, target_sr = 16000):
# Add the visualization part
        # Format: (time, mel)
        plt.figure(figsize = (10, 4))
        librosa.display.specshow(log_S, sr = target_sr, x_axis = 'time', y_axis = 'mel', fmax= target_sr//2)
        plt.colorbar(format = "%+2.0f dB")
        # plt.tight_layout()
        plt.show()

# Test the function to preprocess data
print("Now conduct testing part!")
mel_features = preprocess_audio(file_path)
if mel_features is not None:
    print(f"Feature shape (n_mels, time_steps): {mel_features.shape}")
visualization_mel_spectrogram(mel_features)

# Preprocess all the selected audio files
# Set storage path
print("-------------------------------------------------------------")
print("Now process selected files!")
AUDIO_ROOT = Path("D:/Petra/UrbanSound8K/UrbanSound8K/total_mel_dataset/fold1/mix_0020_1_1.npy")
METADATA_CSV = Path("D:/Petra/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv")
SAVE_DIR = Path("D:/Petra/UrbanSound8K/UrbanSound8K/total_mel_dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES = [1, 3, 4, 5, 8, 9]
df = pd.read_csv(METADATA_CSV)
df_selected = df[df['classID'].isin(SELECTED_CLASSES)].copy()   #df means extract
print(f"Totally, {len(df_selected)} files need to be process!")

processed_info = []

for index, row in tqdm(df_selected.iterrows(), total = df_selected.shape[0]):
    fold_name = f"fold{row['fold']}"
    file_path = AUDIO_ROOT/fold_name/row['slice_file_name']
    if not file_path.exists():
        print(f"File is not exist: {file_path}")
        continue

    mel_spec = preprocess_audio(str(file_path))

    if mel_spec is not None:
        fold_save = SAVE_DIR / fold_name
        fold_save.mkdir(parents=True, exist_ok=True)

        npy_filename = f"{Path(file_path).stem}.npy"
        npy_path = fold_save / npy_filename
        np.save(npy_path, mel_spec)

        row_dict = row.to_dict()
        row_dict['npy_path'] = str(npy_path)
        processed_info.append(row_dict)

df_final = pd.DataFrame(processed_info)
df_final.to_csv("processed_urban8k_train.csv", index=False)
print("Process Successfully!")





