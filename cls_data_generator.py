#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(self, params, split=1, shuffle=True, per_file=False, is_eval=False):

        self._per_file = per_file
        self._is_eval = is_eval

        _arr = np.array(split).astype(int).reshape(-1)  # ex) [[1]] -> [1]
        self._splits = set(_arr.tolist())

        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._multi_accdoa = params.get('multi_accdoa', False)
        self._modality = params.get('modality', 'audio')

        # ✅ 먼저 컨테이너/기본값을 초기화
        self._filenames_list = []        # 파일 베이스명 리스트 (fold1_00000.npy ...)
        self._filelist = []              # 필요 시 풀경로 리스트에 쓸 수 있음
        self._feat_shape = None
        self._label_shape = None

        # ✅ 클래스 수 미리 확보
        self._nb_classes = self._feat_cls.get_nb_classes()

        # ✅ 비디오 모달리티 기본값들
        self._vid_feature_seq_len = params.get('vid_feature_sequence_length', self._label_seq_len)
        if self._modality == 'audio_visual':
            self._vid_feat_dir = self._feat_cls.get_vid_feat_dir()

        self._label_dir = self._feat_cls.get_label_dir()


        norm_dir = self._feat_cls.get_normalized_feat_dir()
        raw_dir  = self._feat_cls.get_unnormalized_feat_dir()

        def _count_npys(p):
            return (os.path.isdir(p) and sum(1 for f in os.listdir(p) if f.endswith('.npy')))

        cnt_norm = _count_npys(norm_dir)
        cnt_raw  = _count_npys(raw_dir)

        if cnt_norm > 0:
            self._feat_dir = norm_dir
            print(f"[DataGen] Using NORMALIZED features: {self._feat_dir} (files={cnt_norm})")
        elif cnt_raw > 0:
            self._feat_dir = raw_dir
            print(f"[DataGen] Using RAW features: {self._feat_dir} (files={cnt_raw})")
        else:
            raise FileNotFoundError(
                "No feature files found.\n"
                f"  - tried norm: {norm_dir}\n"
                f"  - tried raw : {raw_dir}\n"
                f"Check your feat_label_dir / extraction."
            )

        # 라벨 폴더 상태도 알려주기
        cnt_lab = _count_npys(self._label_dir)
        print(f"[DataGen] label_dir={self._label_dir} (files={cnt_lab})")

        self._get_filenames_list_and_feat_label_sizes()


        if self._multi_accdoa:
            label_info = f"{getattr(self, '_num_track_dummy', '?')}x{getattr(self, '_num_axis', '?')}x{getattr(self, '_num_class', self._nb_classes)}"
        else:
            label_info = str(getattr(self, '_label_len', '?'))

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, label_info
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
            feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
            
            # Multi-ACCDOA 출력 크기 조정: SED(1) + X,Y,Z(3) + 추가정보(1, 예: Distance 또는 더미) = 5개 축
            # 기존: self._nb_classes*3*4 -> 클래스당 (활동 + XYZ) * 4 트랙으로 해석되었을 수 있음 (2D 출력 포맷)
            # 하지만 모델은 5D (B, T, 6, 5, 13)를 기대하므로, 2D 출력 형태를 5D로 재해석해야 함.
            # 여기서는 모델이 예상하는 2D 출력 크기에 맞춰 수정합니다. (3 * 4 -> 3 * 5)

            if self._is_eval:
                # Multi-ACCDOA는 레이블 길이가 클래스 * 트랙 * 축 이므로, 3*5로 변경
                label_len_2d = self._nb_classes * 3 * 5 # (Class * 3축 * 5트랙) 또는 (Class * 5축 * 3트랙) 등
                label_shape = (self._batch_size, self._label_seq_len, label_len_2d)
            else:
                if self._multi_accdoa is True:
                    # 🚨 Multi-ACCDOA: 2D 출력 대신 5D 텐서를 직접 생성하거나, 2D 출력의 크기를 5축에 맞게 조정해야 합니다.
                    # 그러나 이 함수는 보통 2D 출력 크기를 정의하므로, 모델이 기대하는 5축을 반영하여 4 -> 5로 변경합니다.
                    label_len_2d = self._nb_classes * 3 * 5 
                    label_shape = (self._batch_size, self._label_seq_len, label_len_2d)
                else:
                    # Single ACCDOA: SED(1) + XYZ(3) = 4축
                    label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*4)

            if self._modality == 'audio_visual':
                vid_feat_shape = (self._batch_size, self._vid_feature_seq_len, 7, 7)
                return feat_shape, vid_feat_shape, label_shape
            return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')

        max_frames, total_frames = -1, 0
        last_feat = None

        # 1) 먼저 split 조건을 만족하는 feature 후보 모으기
        candidates = []
        for filename in os.listdir(self._feat_dir):
            if not (filename.endswith('.npy') and filename.startswith('fold') and len(filename) >= 6 and filename[4].isdigit()):
                continue
            fold_id = int(filename[4])

            if self._is_eval:
                # eval 모드: split 무시
                if self._modality == 'audio' or (hasattr(self, '_vid_feat_dir') and os.path.exists(os.path.join(self._vid_feat_dir, filename))):
                    candidates.append(filename)
            else:
                if fold_id in self._splits:
                    if self._modality == 'audio' or (hasattr(self, '_vid_feat_dir') and os.path.exists(os.path.join(self._vid_feat_dir, filename))):
                        candidates.append(filename)

        if not candidates:
            raise RuntimeError(f"No candidate feature files for splits={sorted(self._splits)} in {self._feat_dir}")

        # 2) (학습/검증 시) 라벨이 "비어있지 않은" 파일만 화이트리스트로 허용
        if not self._is_eval:
            whitelist = []
            removed = 0
            for filename in candidates:
                label_path = os.path.join(self._label_dir, filename)
                if os.path.isfile(label_path):
                    whitelist.append(filename)
                else:
                    removed += 1

            if not whitelist:
                raise RuntimeError(
                    f"All candidate labels were empty or missing. label_dir={self._label_dir}, "
                    f"candidates={len(candidates)}"
                )

            self._filenames_list = sorted(whitelist)
            if removed:
                print(f"[DataGen] filtered out {removed} empty/missing-label files from {len(candidates)} candidates.")
        else:
            # eval 모드에서는 필터링하지 않음 (평가 전량 사용)
            self._filenames_list = sorted(candidates)

        # 3) feat/frames 통계 계산
        for filename in self._filenames_list:
            feat_path = os.path.join(self._feat_dir, filename)
            temp_feat = np.load(feat_path)
            last_feat = temp_feat
            total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
            if temp_feat.shape[0] > max_frames:
                max_frames = temp_feat.shape[0]

        if not self._filenames_list or last_feat is None:
            raise RuntimeError(
                f"Loading features failed — no usable files remain after filtering for splits {sorted(self._splits)} "
                f"in feat_dir={self._feat_dir}"
            )

        # 4) 피처/라벨 모양 파악
        self._nb_frames_file = max_frames if self._per_file else last_feat.shape[0]
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = last_feat.shape[1] // self._nb_mel_bins

        if not self._is_eval:
            first_label = os.path.join(self._label_dir, self._filenames_list[0])
            if not os.path.isfile(first_label):
                raise RuntimeError(f"Label file missing for {self._filenames_list[0]} at {self._label_dir}")

            temp_label = np.load(first_label)
            self._label_shape = temp_label.shape
            if self._multi_accdoa is True:
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3  # Cartesian

        # 5) 배치/총 배치 수
        if self._per_file:
            self._batch_size = int(np.ceil(max_frames / float(self._feature_seq_len)))
            print('\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size * self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len

        if self._modality == 'audio_visual':
            self._vid_feature_batch_seq_len = self._batch_size * self._vid_feature_seq_len

        print(f"[DataGen] usable files: {len(self._filenames_list)} (split={sorted(self._splits)})")
        return


    def generate(self):
        """
        Generates batches of samples
        Returns (training):
            - audio only : feat, label, batch_filenames(list[str])
            - audio_visual: feat, vid_feat, label, batch_filenames(list[str])
        Returns (eval):
            - audio only : feat, batch_filenames
            - audio_visual: feat, vid_feat, batch_filenames
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # frame-level circular buffers
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()
        self._circ_buf_fnames = deque()   # ★ 프레임마다 소속 파일명 병렬 저장

        if self._modality == 'audio_visual':
            self._circ_buf_vid_feat = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # refill circular buffer
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or
                    (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    basename = self._filenames_list[file_cnt]
                    temp_feat = np.load(os.path.join(self._feat_dir, basename))

                    for _row in temp_feat:
                        self._circ_buf_feat.append(_row)
                        self._circ_buf_fnames.append(basename)   # ★ 파일명 기록

                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, basename))
                        for _vf in temp_vid_feat:
                            self._circ_buf_vid_feat.append(_vf)

                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6
                        for _row in extra_feat:
                            self._circ_buf_feat.append(_row)
                            self._circ_buf_fnames.append("PAD")  # ★ 패딩 구분

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones((vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6
                            for _vf in extra_vid_feat:
                                self._circ_buf_vid_feat.append(_vf)

                    file_cnt += 1

                # pop one batch (feature + filenames per-frame)
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                batch_fn_frames = []  # 프레임 단위 파일명
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                    batch_fn_frames.append(self._circ_buf_fnames.popleft())

                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                # 배치 파일명(집합)으로 정리
                batch_filenames = sorted(list({f for f in batch_fn_frames if f != "PAD"}))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)
                    yield feat, vid_feat, batch_filenames
                else:
                    yield feat, batch_filenames

        else:
            for i in range(self._nb_total_batches):
                # refill buffers
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or
                    (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    basename = self._filenames_list[file_cnt]
                    temp_feat = np.load(os.path.join(self._feat_dir, basename))
                    temp_label = np.load(os.path.join(self._label_dir, basename))
                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, basename))

                    if not self._per_file:
                        # align to seq multiples
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]
                        if self._modality == 'audio_visual':
                            temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len, :, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                        self._circ_buf_fnames.append(basename)  # ★

                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    if self._modality == 'audio_visual':
                        for vf_row in temp_vid_feat:
                            self._circ_buf_vid_feat.append(vf_row)

                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones((vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros((label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                            self._circ_buf_fnames.append("PAD")   # ★
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)
                        if self._modality == 'audio_visual':
                            for vf_row in extra_vid_feat:
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt += 1

                # pop one batch (feat/label + filenames per frame)
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                batch_fn_frames = []
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                    batch_fn_frames.append(self._circ_buf_fnames.popleft())
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()

                if self._multi_accdoa is True:
                    label = np.zeros((self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()

                # split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))
                if self._modality == 'audio_visual':
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

                label = self._split_in_seqs(label, self._label_seq_len)

                if self._multi_accdoa is True:
                    pass
                else:
                    # SED mask broadcast 후 ACCDOA/REG 레이블에 적용
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 4)
                    label = mask * label[:, :, self._nb_classes:]

                # ★ 프레임 단위 파일명 → 배치에 참여한 파일명 set
                batch_filenames = sorted(list({f for f in batch_fn_frames if f != "PAD"}))

                if self._modality == 'audio_visual':
                    yield feat, vid_feat, label, batch_filenames
                else:
                    yield feat, label, batch_filenames

    def _split_in_seqs(self, data, _seq_len): # data - 250*8, 7, 64 - 250
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def _vid_feat_split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            else:
                data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions for video features: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
