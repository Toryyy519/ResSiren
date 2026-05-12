import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import os
from pathlib import Path
from tqdm import tqdm
from audiomentations import AddBackgroundNoise, Normalize, Compose


class AudioMixer:
    def __init__(self, csv_path, audio_root, noise_root, output_npy_dir, wav_output_dir, sr=16000, duration=4.0):
        self.df = pd.read_csv(csv_path)
        self.audio_root = Path(audio_root)
        self.noise_root = noise_root
        self.output_npy_dir = Path(output_npy_dir)
        self.wav_output_dir = Path(wav_output_dir)
        self.sr = sr
        self.duration = duration
        self.target_samples = int(self.sr * self.duration)

        self.output_npy_dir.mkdir(parents=True, exist_ok=True)
        self.wav_output_dir.mkdir(parents=True, exist_ok=True)

    def get_label(self, class_id_list):
        if 1 in class_id_list:
            return 1
        elif 8 in class_id_list:
            return 2
        else:
            return 3

    def load_audio(self, path):
        try:
            audio, _ = librosa.load(path, sr=self.sr)
            if len(audio) < self.target_samples:
                padding_needed = self.target_samples - len(audio)
                audio = np.pad(audio, (0, padding_needed), mode='constant')
            else:
                audio = audio[:self.target_samples]
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.target_samples, dtype=np.float32)

    def get_augmentation_pipeline(self):
        return Compose([
            AddBackgroundNoise(
                sounds_path=self.noise_root,
                min_snr_db=10.0,
                max_snr_db=25.0,
                p=0.6
            ),
            Normalize(p=1.0)
        ])

    def synthesis(self, total_count_per_fold):
        results = []
        mix_options = [2, 3, 4, 5]
        mix_probs = [0.4, 0.3, 0.2, 0.1]
        pipeline = self.get_augmentation_pipeline()

        all_folds = sorted(self.df['fold'].unique())

        for fold_id in all_folds:
            print(f"\nProcessing Fold {fold_id}...")
            fold_df = self.df[self.df['fold'] == fold_id]

            fold_npy_path = self.output_npy_dir / f"fold{fold_id}"
            fold_wav_path = self.wav_output_dir / f"fold{fold_id}"
            fold_npy_path.mkdir(parents=True, exist_ok=True)
            fold_wav_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(total_count_per_fold), desc=f"Synthesizing Fold {fold_id}"):
                num_to_mix = np.random.choice(mix_options, p=mix_probs)
                batch_samples = fold_df.sample(n=num_to_mix, replace=True)

                mixed_audio = np.zeros(self.target_samples, dtype=np.float32)
                class_ids = []

                for _, row in batch_samples.iterrows():
                    file_path = self.audio_root / f"fold{row['fold']}" / row['slice_file_name']
                    audio_segment = self.load_audio(file_path)

                    gain = np.random.uniform(0.5, 0.9)
                    mixed_audio += audio_segment * gain
                    class_ids.append(int(row['classID']))

                final_audio = pipeline(samples=mixed_audio, sample_rate=self.sr)
                final_label = self.get_label(class_ids)
                filename = f"mix_{i:04d}_{fold_id}_{final_label}"

                wav_file = fold_wav_path / f"{filename}.wav"
                sf.write(str(wav_file), final_audio, self.sr)

                mel_spec = librosa.feature.melspectrogram(
                    y=final_audio,
                    sr=self.sr,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=128
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                npy_file = fold_npy_path / f"{filename}.npy"
                np.save(str(npy_file), log_mel_spec)

                results.append({
                    "file_name": f"{filename}.npy",
                    "fold": fold_id,
                    "label": final_label,
                    "mixed_classes": class_ids
                })

        log_df = pd.DataFrame(results)
        log_df.to_csv("synthesis_metadata_log.csv", index=False)
        print("\nSynthesis process completed successfully.")


if __name__ == "__main__":
    mixer = AudioMixer(
        csv_path=r"C:\Users\Petra\Desktop\ujm\projects\Code\processed_urban8k_train.csv",
        audio_root=r"D:\Petra\UrbanSound8K\UrbanSound8K\audio",
        noise_root=r"D:\Petra\UrbanSound8K\UrbanSound8K\noise",
        output_npy_dir=r"D:\Petra\UrbanSound8K\UrbanSound8K\total_mel_dataset",
        wav_output_dir=r"D:\Petra\UrbanSound8K\UrbanSound8K\mixoutput_wave"
    )

    # Generate 1000 samples per fold
    mixer.synthesis(total_count_per_fold=1000)