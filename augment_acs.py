import os
import glob
import numpy as np
import pandas as pd
import torchaudio
from rich.progress import Progress

def augment_acs(params):
    """
    Applies 8-fold Audio Channel Swap (ACS) augmentation to FOA audio and metadata.
    """
    # 원본 데이터 경로
    dev_audio_path = os.path.join(params['dataset_dir'], 'foa_dev')
    dev_meta_path = os.path.join(params['dataset_dir'], 'metadata_dev', 'dev-train') # DCASE 2024 train split
    
    # 증강 데이터를 저장할 새로운 경로
    aug_audio_path = os.path.join(params['dataset_dir'], 'foa_dev_aug')
    aug_meta_path = os.path.join(params['dataset_dir'], 'metadata_dev', 'dev-train-aug')
    os.makedirs(aug_audio_path, exist_ok=True)
    os.makedirs(aug_meta_path, exist_ok=True)

    audio_files = sorted(glob.glob(os.path.join(dev_audio_path, '*.wav')))
    
    # 8가지 채널 변환 매트릭스 정의
    # 순서: W, X, Y, Z
    channel_transforms = [
        [1, 1, 1, 1],    # 1. Original
        [1, 1, -1, 1],   # 2. Flip Y
        [1, 1, 1, -1],   # 3. Flip Z
        [1, 1, -1, -1],  # 4. Flip Y, Z
        [1, -1, 1, 1],   # 5. Flip X
        [1, -1, -1, 1],  # 6. Flip X, Y
        [1, -1, 1, -1],  # 7. Flip X, Z
        [1, -1, -1, -1]  # 8. Flip X, Y, Z
    ]

    with Progress() as progress:
        task = progress.add_task("[cyan]Applying ACS Augmentation...", total=len(audio_files))
        for audio_file in audio_files:
            filename_base = os.path.splitext(os.path.basename(audio_file))[0]
            label_file = os.path.join(dev_meta_path, filename_base + '.csv')

            if not os.path.exists(label_file):
                progress.update(task, advance=1)
                continue

            waveform, sr = torchaudio.load(audio_file)
            labels_df = pd.read_csv(label_file, header=None)

            for i, transform in enumerate(channel_transforms):
                aug_suffix = f'_acs{i}'
                
                # 새로운 파일 경로
                new_audio_file = os.path.join(aug_audio_path, filename_base + aug_suffix + '.wav')
                new_label_file = os.path.join(aug_meta_path, filename_base + aug_suffix + '.csv')
                
                # 1. 오디오 변환 및 저장
                new_waveform = waveform.clone()
                new_waveform[1, :] *= transform[1] # X channel
                new_waveform[2, :] *= transform[2] # Y channel
                new_waveform[3, :] *= transform[3] # Z channel
                torchaudio.save(new_audio_file, new_waveform, sr)

                # 2. 라벨 변환 및 저장
                new_labels_df = labels_df.copy()
                azimuth = new_labels_df.iloc[:, 3]
                elevation = new_labels_df.iloc[:, 4]

                # 각 변환에 맞는 각도 계산
                if transform[1] == -1: # Flip X
                    azimuth = 180 - azimuth
                    # -180 ~ 180 범위 유지
                    azimuth[azimuth > 180] -= 360
                if transform[2] == -1: # Flip Y
                    azimuth = -azimuth
                if transform[3] == -1: # Flip Z
                    elevation = -elevation
                
                new_labels_df.iloc[:, 3] = azimuth
                new_labels_df.iloc[:, 4] = elevation
                new_labels_df.to_csv(new_label_file, header=False, index=False)
            
            progress.update(task, advance=1)
            
    print(f"\nACS augmentation completed!")
    print(f"Augmented audio saved to: {aug_audio_path}")
    print(f"Augmented metadata saved to: {aug_meta_path}")


if __name__ == '__main__':
    # 이 스크립트를 실행하기 전에 parameters.py를 수정하여 dataset_dir을 정확히 설정해야 합니다.
    import parameters
    params = parameters.get_params('3') # 기본 파라미터 로드
    augment_acs(params)