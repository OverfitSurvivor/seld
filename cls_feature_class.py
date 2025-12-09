# Contains routines for labels creation, features extraction and normalization
#

from cls_vid_features import VideoFeatures
from PIL import Image
import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib
import cv2
import soundfile as sf
import pandas as pd
import re

def _read_starss23_synth_labels(csv_path):
    df = pd.read_csv(csv_path)
    cols = ["frame", "class_idx", "source_idx", "azimuth", "elevation", "distance_cm"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df[cols]

def _is_valid_foa_file(path, expected_channels):
    try:
        info = sf.info(path)
        if info.channels != expected_channels:
            return False
        return True
    except Exception:
        return False

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)
        
        if is_eval:
            self._desc_dir = None
        else:
            self._desc_dir = os.path.join(self._dataset_dir, 'metadata_dev')

        self._vid_dir = os.path.join(self._dataset_dir, 'video_{}'.format('eval' if is_eval else 'dev'))
        
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None
        self._vid_feat_dir = None

        self._is_eval = is_eval
        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)
        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params.get('use_salsalite', False)
        self._use_stpacc = params.get('use_stpacc', False)

        if self._use_salsalite and self._dataset=='mic':
            self._lower_bin = np.int(np.floor(params['fmin_doa_salsalite'] * self._nfft / np.float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = np.int(np.floor(np.min((params['fmax_doa_salsalite'], self._fs//2)) * self._nfft / np.float(self._fs)))
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft//2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]
            self._cutoff_bin = np.int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / np.float(self._fs)))
            self._nb_mel_bins = self._cutoff_bin - self._lower_bin
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        
        self._nb_unique_classes = params['unique_classes']
        self._filewise_frames = {}

    def get_frame_stats(self):
        """
        Compute frame counts from available files.
        Priority:
        1. WAV files (if available)
        2. Features in 'foa_dev' (unnormalized)
        3. Features in 'foa_dev_norm' (normalized)
        """
        if len(self._filewise_frames) != 0:
            return
        
        print(f"[DBG] Computing frame stats...")
        
        # 1. Try finding WAV files
        wav_files = []
        if os.path.exists(self._aud_dir):
            for r, _, files in os.walk(self._aud_dir):
                for f in files:
                    if f.lower().endswith('.wav'):
                        wav_files.append(os.path.join(r, f))
        
        if len(wav_files) > 0:
            print(f"[DBG] Found {len(wav_files)} WAV files. Using WAV for stats.")
            for wav_path in sorted(wav_files):
                if not _is_valid_foa_file(wav_path, self._nb_channels):
                    continue
                base = os.path.basename(wav_path).rsplit('.', 1)[0]
                info = sf.info(wav_path)
                audio_len = info.frames
                nb_feat_frames  = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[base] = [nb_feat_frames, nb_label_frames]
        else:
            # 2. Fallback: Try Feature Dirs (foa_dev or foa_dev_norm)
            feat_dirs = [self.get_unnormalized_feat_dir(), self.get_normalized_feat_dir()]
            found_features = False
            
            for f_dir in feat_dirs:
                if os.path.exists(f_dir):
                    npy_files = [f for f in os.listdir(f_dir) if f.endswith('.npy')]
                    if len(npy_files) > 0:
                        print(f"[DBG] Found {len(npy_files)} .npy files in {f_dir}. Using features for stats.")
                        
                        ratio = self._label_hop_len / self._hop_len # typically 5.0
                        
                        for f in sorted(npy_files):
                            base = f.split('.')[0]
                            path = os.path.join(f_dir, f)
                            try:
                                # mmap_mode='r' reads only metadata (shape) quickly
                                feat = np.load(path, mmap_mode='r')
                                nb_feat_frames = feat.shape[0]
                                nb_label_frames = int(nb_feat_frames / ratio)
                                self._filewise_frames[base] = [nb_feat_frames, nb_label_frames]
                            except Exception as e:
                                print(f"[WARN] Failed to read {f}: {e}")
                        
                        found_features = True
                        break # Found valid features, stop checking other dirs
            
            if not found_features:
                print(f"[ERROR] No WAVs and No Feature files found! Cannot calculate frame stats.")
                print(f"Checked: {self._aud_dir}")
                print(f"Checked: {feat_dirs}")

        print(f"[DBG] Frame stats computed for {len(self._filewise_frames)} valid files.")

    def _load_audio(self, audio_filename):
        audio, fs = sf.read(audio_filename, dtype='float32', always_2d=True)
        if audio.shape[1] < self._nb_channels:
            raise ValueError(f"Invalid channel count {audio.shape[1]} for {audio_filename}")
        audio = audio[:, :self._nb_channels]
        audio = audio + self._eps
        return audio, fs

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T 

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1)) / 3.0)
        I_norm = I / E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self._mel_wts), (0, 2, 1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        return foa_iv

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return np.concatenate((linear_spectra, phase_vector), axis=-1)

    def _get_stp_acc(self, linear_spectra):
        R = np.abs(linear_spectra)**2 
        n_frames = R.shape[0]
        n_ch = R.shape[2]
        
        acc_len = self._nfft // 2
        
        stp_feat = np.zeros((n_frames, self._nb_mel_bins, n_ch))
        hanning_win = np.hanning(8)

        for ch in range(n_ch):
            spec_ch = R[:, :, ch]
            acc = np.fft.irfft(spec_ch, n=self._nfft, axis=1)
            acc = acc[:, :acc_len]
            acc = np.clip(acc, 1e-100, None)
            
            max_acc = np.max(np.abs(acc), axis=1, keepdims=True)
            max_acc[max_acc == 0] = 1e-8
            acc = acc / max_acc
            
            acc_sq = acc**2
            
            for i in range(n_frames):
                stp = np.convolve(acc_sq[i], hanning_win, 'same')
                mx = np.max(stp)
                if mx == 0: mx = 1e-8
                stp = stp / mx
                stp_ds = scipy.signal.resample(stp, self._nb_mel_bins)
                stp_feat[i, :, ch] = stp_ds

        return stp_feat.transpose((0, 2, 1)).reshape((n_frames, -1))

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)
        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]
        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec

    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
                    dist_label[frame_ind, active_event[0]] = active_event[5]

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)
        return label_mat

    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:
                        self._fill_adpit_labels(active_event_list_per_class, frame_ind, se_label, x_label, y_label, z_label, dist_label)
                    elif active_event[0] != active_event_list[i + 1][0]:
                        self._fill_adpit_labels(active_event_list_per_class, frame_ind, se_label, x_label, y_label, z_label, dist_label)
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)
        return label_mat

    def _fill_adpit_labels(self, event_list, frame_ind, se, x, y, z, dist):
        n_events = len(event_list)
        if n_events == 1:
            e = event_list[0]; idx=0
            se[frame_ind, idx, e[0]] = 1
            x[frame_ind, idx, e[0]] = e[2]; y[frame_ind, idx, e[0]] = e[3]; z[frame_ind, idx, e[0]] = e[4]; dist[frame_ind, idx, e[0]] = e[5]
        elif n_events == 2:
            e0 = event_list[0]; e1 = event_list[1]
            se[frame_ind, 1, e0[0]] = 1; x[frame_ind, 1, e0[0]] = e0[2]; y[frame_ind, 1, e0[0]] = e0[3]; z[frame_ind, 1, e0[0]] = e0[4]; dist[frame_ind, 1, e0[0]] = e0[5]
            se[frame_ind, 2, e1[0]] = 1; x[frame_ind, 2, e1[0]] = e1[2]; y[frame_ind, 2, e1[0]] = e1[3]; z[frame_ind, 2, e1[0]] = e1[4]; dist[frame_ind, 2, e1[0]] = e1[5]
        else: # >=3
            for k in range(3):
                if k < n_events:
                    e = event_list[k]
                    se[frame_ind, 3+k, e[0]] = 1
                    x[frame_ind, 3+k, e[0]] = e[2]; y[frame_ind, 3+k, e[0]] = e[3]; z[frame_ind, 3+k, e[0]] = e[4]; dist[frame_ind, 3+k, e[0]] = e[5]

    def extract_file_feature(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in
        spect = self._get_spectrogram_for_file(_wav_path)

        mel_spect = None
        if not self._use_salsalite:
            mel_spect = self._get_mel_spectrogram(spect)

        feat = None
        if self._dataset == 'foa':
            if self._use_stpacc: 
                stp_acc = self._get_stp_acc(spect)
                foa_iv = self._get_foa_intensity_vectors(spect)
                feat = np.concatenate((mel_spect, stp_acc, foa_iv), axis=-1)
            else:
                foa_iv = self._get_foa_intensity_vectors(spect)
                feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self._dataset == 'mic':
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            print('ERROR: Unknown dataset format {}'.format(self._dataset))
            exit()

        if feat is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feat.shape))
            np.save(_feat_path, feat)

    def extract_all_feature(self):
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)
        import time
        start_s = time.time()

        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        def _collect_wavs(root):
            wavs = []
            for r, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith('.wav'):
                        wavs.append(os.path.join(r, f))
            return sorted(wavs)

        wav_files = _collect_wavs(self._aud_dir)
        for file_cnt, wav_path in enumerate(wav_files):
            if not _is_valid_foa_file(wav_path, self._nb_channels):
                continue
            feat_path = os.path.join(self._feat_dir, os.path.basename(wav_path).replace('.wav', '.npy'))
            self.extract_file_feature((file_cnt, wav_path, feat_path))
        print(time.time() - start_s)

    def preprocess_features(self):
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))
        else:
            print('Estimating weights for normalizing feature files:')
            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(spec_scaler, normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(os.path.join(self._feat_dir_norm, file_name), feat_file)
            del feat_file

    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()
        
        print('\n[EXTRACT ALL LABELS] Start')
        print(f'[DBG] Label Dir: {self._label_dir}')
        print(f'[DBG] Metadata Dir (desc_dir): {self._desc_dir}')
        
        create_folder(self._label_dir)
        valid_wav_bases = set(self._filewise_frames.keys())
        
        print(f"[DBG] Valid wav/npy files in memory: {len(valid_wav_bases)}")
        
        if not os.path.exists(self._desc_dir):
            print(f"[ERROR] Metadata directory NOT FOUND: {self._desc_dir}")
            print(f"Please check 'dataset_dir' in parameters.py and ensure 'metadata_dev' exists.")
            return

        total_csv_found = 0
        total_matched = 0
        
        for root, dirs, files in os.walk(self._desc_dir):
            desc_files = [f for f in files if f.lower().endswith('.csv')]
            total_csv_found += len(desc_files)
            
            for file_name in desc_files:
                base = os.path.splitext(file_name)[0]
                if base not in valid_wav_bases: 
                    # print(f"[SKIP] No matching wav for {file_name}")
                    continue
                
                total_matched += 1
                nb_feat_frames, nb_label_frames = self._filewise_frames[base]
                desc_file_polar = self.load_output_format_file(os.path.join(root, file_name), cm2m=True)
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                if self._multi_accdoa:
                    label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                else:
                    label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                np.save(os.path.join(self._label_dir, f'{base}.npy'), label_mat)
        
        print(f"[DBG] Total CSVs found: {total_csv_found}")
        print(f"[DBG] Total Labels Created: {total_matched}")
        print('[EXTRACT ALL LABELS] Done\n')

    # ------------------------------- VISUAL FEATURES -------------------------------
    @staticmethod
    def _read_vid_frames(vid_filename):
        cap = cv2.VideoCapture(vid_filename)
        pil_frames = []
        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_cnt % 3 == 0:
                resized_frame = cv2.resize(frame, (360, 180))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)
            frame_cnt += 1
        cap.release()
        return pil_frames

    def extract_file_vid_feature(self, _arg_in):
        _file_cnt, _mp4_path, _vid_feat_path = _arg_in
        vid_frames = self._read_vid_frames(_mp4_path)
        pretrained_vid_model = VideoFeatures()
        vid_feat = pretrained_vid_model(vid_frames)
        vid_feat = np.array(vid_feat)
        if vid_feat is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_mp4_path), vid_feat.shape))
            np.save(_vid_feat_path, vid_feat)

    def extract_visual_features(self):
        self._vid_feat_dir = self.get_vid_feat_dir()
        create_folder(self._vid_feat_dir)
        print('Extracting visual features:')
        for sub_folder in os.listdir(self._vid_dir):
            loc_vid_folder = os.path.join(self._vid_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_vid_folder)):
                mp4_filename = '{}.mp4'.format(file_name.split('.')[0])
                mp4_path = os.path.join(loc_vid_folder, mp4_filename)
                vid_feat_path = os.path.join(self._vid_feat_dir, '{}.npy'.format(mp4_filename.split('.')[0]))
                self.extract_file_vid_feature((file_cnt, mp4_path, vid_feat_path))

    # ------------------------------- DCASE IO -------------------------------
    def load_output_format_file(self, _output_format_file, cm2m=False):
        _output_dict = {}
        with open(_output_format_file, 'r', encoding='utf-8') as _fid:
            pos = _fid.tell()
            first = _fid.readline()
            if not first: return _output_dict
            if len(first.strip().split(',')) == 0 or first.strip().split(',')[0].strip().lower() != 'frame':
                _fid.seek(pos)
            
            _words = []
            for _line in _fid:
                _line = _line.strip()
                if not _line: continue
                _words = _line.split(',')
                try: _frame_ind = int(_words[0])
                except ValueError: continue
                if _frame_ind not in _output_dict: _output_dict[_frame_ind] = []
                
                if len(_words) == 4:
                    _output_dict[_frame_ind].append([int(_words[1]), 0, float(_words[2]), float(_words[3])])
                elif len(_words) == 5:
                    _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
                elif len(_words) == 6:
                    _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])/100 if cm2m else float(_words[5])])
                elif len(_words) == 7:
                    _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])/100 if cm2m else float(_words[6])])
        
        if len(_words) == 7:
            _output_dict = self.convert_output_format_cartesian_to_polar(_output_dict)
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        _fid = open(_output_format_file, 'w')
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict: continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict: loc_dict[value[0]] = {}
                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]: loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]: output_dict[block_cnt][class_cnt] = []
                keys = [k for k in loc_dict[class_cnt]]; values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]
                output_dict[block_cnt][class_cnt].append([keys, values])
        return output_dict

    def organize_labels(self, _pred_dict, _max_frames):
        nb_frames = _max_frames
        output_dict = {x: {} for x in range(nb_frames)}
        for frame_idx in range(0, _max_frames):
            if frame_idx not in _pred_dict: continue
            for [class_idx, track_idx, *localization] in _pred_dict[frame_idx]:
                if class_idx not in output_dict[frame_idx]: output_dict[frame_idx][class_idx] = {}
                if track_idx not in output_dict[frame_idx][class_idx]: output_dict[frame_idx][class_idx][track_idx] = localization
                else:
                    min_track_idx = np.min(np.array(list(output_dict[frame_idx][class_idx].keys())))
                    new_track_idx = min_track_idx - 1 if min_track_idx < 0 else -1
                    output_dict[frame_idx][class_idx][new_track_idx] = localization
        return output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict: out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                ele_rad = tmp_val[3]*np.pi/180.; azi_rad = tmp_val[2]*np.pi/180.
                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label; y = np.sin(azi_rad) * tmp_label; z = np.sin(ele_rad)
                out_dict[frame_cnt].append(tmp_val[0:2] + [x, y, z] + tmp_val[4:])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict: out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                out_dict[frame_cnt].append(tmp_val[0:2] + [azimuth, elevation] + tmp_val[5:])
        return out_dict

    def get_unnormalized_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'foa_dev')

    def get_normalized_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'foa_dev_norm')

    def get_label_dir(self):
        cand_a = os.path.join(self._feat_label_dir, 'foa_dev_adpit_label')
        cand_b = os.path.join(self._feat_label_dir, 'foa_dev_label')
        for p in (cand_a, cand_b):
            if os.path.isdir(p) and any(f.endswith('.npy') for f in os.listdir(p)): return p
        return cand_a

    def get_normalized_wts_file(self):
        return os.path.join(self._feat_label_dir, '{}_wts'.format(self._dataset))

    def get_vid_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'video_{}'.format('eval' if self._is_eval else 'dev'))

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)