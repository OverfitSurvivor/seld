#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
# [메모리 최적화 1] OOM 방지를 위한 환경 변수 설정 (가장 먼저 실행되어야 함)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # AMP 사용

plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import resnet_conformer_model as seldnet_model
from tqdm import tqdm
import pandas as pd
from scipy.signal import medfilt
import torch.nn.functional as F
import random

def foa_acs_gain_augment(data, target, params):
    """
    data:   [B, C_in, T, F]
    target: [B, T, 6, 5, C]  (ADPIT: [act, x, y, z, dist])
    """
    if params.get('dataset', 'foa') != 'foa':
        return data, target  # FOA가 아니면 증강 안 함

    B, C_in, T, F = data.shape
    if C_in < 4:
        return data, target  # FOA 4채널(WXYZ)이 없다면 스킵

    device = data.device
    dtype  = data.dtype

    acs_prob  = float(params.get('acs_prob', 0.5))
    gain_prob = float(params.get('gain_prob', 0.5))
    gain_min  = float(params.get('gain_min', 0.7))
    gain_max  = float(params.get('gain_max', 1.3))

    # ---------- 1) ACS: X/Y/Z sign flip ----------
    if random.random() < acs_prob:
        # flips: [sx, sy, sz] 각각 ±1
        flips = torch.ones(3, device=device, dtype=dtype)
        for i in range(3):
            if random.random() < 0.5:
                flips[i] = -1.0

        # feature 쪽: X,Y,Z = data[:,1], data[:,2], data[:,3]
        data[:, 1:4] = data[:, 1:4] * flips.view(1, 3, 1, 1)

        # label 쪽: DOA (x,y,z)에 동일한 flip 적용
        # target shape: [B, T, 6, 5, C], 5축 = [act, x, y, z, dist]
        if target.ndim == 5 and target.size(2) == 6 and target.size(3) == 5:
            # xyz: [B, T, 6, 3, C]
            xyz = target[:, :, :, 1:4, :]
            xyz = xyz * flips.view(1, 1, 1, 3, 1)
            target[:, :, :, 1:4, :] = xyz

    # ---------- 2) Gain 증강 ----------
    if random.random() < gain_prob:
        g = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(gain_min, gain_max)
        data = data * g   # 거리 라벨은 그대로 두는 게 일반적

    return data, target

def hysteresis_1d(prob, low_thr, high_thr):
    """
    prob: 1D np.array in [0,1], shape [T]
    low_thr, high_thr: float
    return: 1D np.array of {0,1} with hysteresis
    """
    T = prob.shape[0]
    active = np.zeros(T, dtype=bool)

    # 1) high_thr로 기본 on/off
    active = prob >= high_thr

    # 2) forward pass: 이전이 active였고 low_thr 이상이면 유지
    for t in range(1, T):
        if not active[t] and active[t-1] and prob[t] >= low_thr:
            active[t] = True

    # 3) backward pass: 다음이 active였고 low_thr 이상이면 유지
    for t in range(T-2, -1, -1):
        if not active[t] and active[t+1] and prob[t] >= low_thr:
            active[t] = True

    return active.astype(np.float32)


def postprocess_sed_probs(sed_probs, params):
    """
    sed_probs: np.array [B, T, tracks, C] in [0,1]
    returns: bin_sed: np.array [B, T, tracks, C] with {0,1}
    """
    B, T, K, C = sed_probs.shape

    # --- 1) median filter (time axis) ---
    kernel = int(params.get('sed_median_kernel', 0))
    if kernel > 1:
        for b in range(B):
            for k in range(K):
                for c in range(C):
                    sed_probs[b, :, k, c] = medfilt(
                        sed_probs[b, :, k, c],
                        kernel_size=kernel
                    )

    # --- 2) single threshold ---
    thr = float(params.get('sed_threshold', 0.3))

    # [B,T,K,C] → {0,1}
    bin_sed = (sed_probs >= thr).astype(np.float32)

    return bin_sed


def decode_tracks(output_dict, nb_classes, sed_thr=None, params=None):
    if params is None:
        params = {}
    thr = float(params.get('sed_threshold', 0.3))
    if sed_thr is None:
        sed_thr = thr

    sed_logits = output_dict["sed"]   # [B,T, C*tracks]
    doa        = output_dict["doa"]   # [B,T, 3*C*tracks]
    dist       = output_dict["dist"]  # [B,T, C*tracks]
    B, T, _    = sed_logits.shape
    tracks     = 3

    # 1) SED 확률
    sed_probs = torch.sigmoid(sed_logits).view(B, T, tracks, nb_classes)   # [B,T,3,C]

    # 2) numpy로 옮겨서 후처리
    sed_probs_np = sed_probs.detach().cpu().numpy()
    sed_np = postprocess_sed_probs(sed_probs_np, params)   # [B,T,3,C] in {0,1}

    # 3) DOA / Dist reshape
    doa  = doa.view(B, T, tracks, nb_classes, 3)   # [B,T,3,C,3]
    dist = dist.view(B, T, tracks, nb_classes)     # [B,T,3,C]

    with torch.no_grad():
        norm = torch.linalg.norm(doa, dim=-1, keepdim=True).clamp_min(1e-8)
        doa  = doa / norm
        doa  = torch.nan_to_num(doa, nan=0.0, posinf=0.0, neginf=0.0)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    doa_np  = doa.detach().cpu().numpy()      # [B,T,3,C,3]
    dist_np = dist.detach().cpu().numpy()     # [B,T,3,C]

    return sed_np, doa_np, dist_np


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def eval_epoch(data_generator, model, dcase_output_folder, params, device):
    thr = float(params.get('sed_threshold', 0.2))
    eval_filelist = data_generator.get_filelist()
    model.eval()
    file_cnt = 0

    with torch.no_grad():
        loop = tqdm(
            data_generator.generate(),
            total=data_generator.get_total_batches_in_data(),
            desc='Validation',
            leave=False,
            ncols=100
        )
        for values in loop:
            if len(values) == 2:
                data, batch_filenames = values
                data = torch.tensor(data).to(device).float()
                output = model(data)
            elif len(values) == 3:
                data, vid_feat, batch_filenames = values
                data = torch.tensor(data).to(device).float()
                vid_feat = torch.tensor(vid_feat).to(device).float()
                output = model(data, vid_feat)
            else:
                raise RuntimeError(f"[EVAL] Unexpected values length={len(values)} from generator")

            if params['multi_accdoa'] is True:
                sed_np, doa_np, dist_np = decode_tracks(
                    output, params['unique_classes'], params=params
                )
                sed_np  = sed_np[0]        # [T,3,C]
                doa_np  = doa_np[0]        # [T,3,C,3]
                dist_np = dist_np[0]       # [T,3,C]

                dist_np = np.nan_to_num(dist_np, nan=0.0, posinf=0.0, neginf=0.0)
                dist_np = np.clip(dist_np, 0.0, 1e4)

                sed_pred0 = sed_np[:, 0, :]   # [T,C]
                sed_pred1 = sed_np[:, 1, :]
                sed_pred2 = sed_np[:, 2, :]

                def pack_doa(track_idx):
                    xyz = doa_np[:, track_idx, :, :]            # [T,C,3]
                    return np.concatenate(
                        [xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]],
                        axis=-1
                    )  # [T,3*C]

                doa_pred0 = pack_doa(0)
                doa_pred1 = pack_doa(1)
                doa_pred2 = pack_doa(2)
                dist_pred0 = dist_np[:, 0, :]   # [T,C]
                dist_pred1 = dist_np[:, 1, :]
                dist_pred2 = dist_np[:, 2, :]

            else:
                sed_logits = output["sed"]
                doa_out    = output["doa"]
                B, T, _    = sed_logits.shape
                sed_np = (torch.sigmoid(sed_logits).detach().cpu().numpy() >= thr).astype(np.float32)
                doa_np = doa_out.detach().cpu().numpy().reshape(B, T, params['unique_classes'], 3)

                sed_np  = sed_np[0]
                doa_np  = doa_np[0]
                sed_pred = sed_np
                doa_pred = np.concatenate(
                    [doa_np[:, :, 0], doa_np[:, :, 1], doa_np[:, :, 2]],
                    axis=-1
                )

            output_file = os.path.join(
                dcase_output_folder,
                eval_filelist[file_cnt].replace('.npy', '.csv')
            )
            file_cnt += 1

            output_dict = {}

            if params['multi_accdoa'] is True:
                T, C = sed_pred0.shape
                for frame_cnt in range(T):
                    for class_cnt in range(C):
                        flag_0sim1 = determine_similar_location(
                            sed_pred0[frame_cnt][class_cnt],
                            sed_pred1[frame_cnt][class_cnt],
                            doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )
                        flag_1sim2 = determine_similar_location(
                            sed_pred1[frame_cnt][class_cnt],
                            sed_pred2[frame_cnt][class_cnt],
                            doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )
                        flag_2sim0 = determine_similar_location(
                            sed_pred2[frame_cnt][class_cnt],
                            sed_pred0[frame_cnt][class_cnt],
                            doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )

                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > thr:
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + params['unique_classes']], doa_pred0[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred0[frame_cnt][class_cnt])])
                            if sed_pred1[frame_cnt][class_cnt] > thr:
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + params['unique_classes']], doa_pred1[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred1[frame_cnt][class_cnt])])
                            if sed_pred2[frame_cnt][class_cnt] > thr:
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + params['unique_classes']], doa_pred2[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred2[frame_cnt][class_cnt])])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > thr:
                                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + params['unique_classes']], doa_pred2[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred2[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                dist_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + params['unique_classes']], doa_fc[class_cnt + 2 * params['unique_classes']], float(dist_fc[class_cnt])])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > thr:
                                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + params['unique_classes']], doa_pred0[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred0[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                dist_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + params['unique_classes']], doa_fc[class_cnt + 2 * params['unique_classes']], float(dist_fc[class_cnt])])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > thr:
                                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + params['unique_classes']], doa_pred1[frame_cnt][class_cnt + 2 * params['unique_classes']], float(dist_pred1[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                dist_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + params['unique_classes']], doa_fc[class_cnt + 2 * params['unique_classes']], float(dist_fc[class_cnt])])
                        else:
                            doa_fc  = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            dist_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + params['unique_classes']], doa_fc[class_cnt + 2 * params['unique_classes']], float(dist_fc[class_cnt])])
            else:
                T, C = sed_pred.shape
                for frame_cnt in range(T):
                    for class_cnt in range(C):
                        if sed_pred[frame_cnt][class_cnt] > thr:
                            output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt + C], doa_pred[frame_cnt][class_cnt + 2 * C]])

            data_generator.write_output_format_file(output_file, output_dict)


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    import glob, os

    thr = float(params.get('sed_threshold', 0.2))
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    saw_first = False

    with torch.no_grad():
        loop = tqdm(
            data_generator.generate(),
            total=data_generator.get_total_batches_in_data(),
            desc='Validation',
            leave=False,
            ncols=100
        )
        for values in loop:
            if len(values) == 3:
                data, label, batch_filenames = values
                data  = torch.tensor(data).to(device).float()
                label = torch.tensor(label).to(device).float()
                output = model(data)
            elif len(values) == 4:
                data, vid_feat, label, batch_filenames = values
                data    = torch.tensor(data).to(device).float()
                vid_feat = torch.tensor(vid_feat).to(device).float()
                label   = torch.tensor(label).to(device).float()
                output  = model(data, vid_feat)
            else:
                raise RuntimeError(f"[VAL] Unexpected values length={len(values)} from generator")

            loss = criterion(output, label)
            # --- 디버그: SED 활성 비율 체크 (한 번만) ---
            if nb_test_batches == 0:  # 첫 배치일 때만
                with torch.no_grad():
                    thr = float(params.get('sed_threshold', 0.1))

                    # 1) 예측 SED
                    sed_logits = output["sed"]  # [B, T, 3*C]
                    B, T, _ = sed_logits.shape
                    tracks = 3
                    C = params['unique_classes']

                    sed_prob = torch.sigmoid(sed_logits).view(B, T, tracks, C)   # [B, T, 3, C]
                    sed_pred_bin = (sed_prob >= thr)                             # bool

                    pred_active_ratio = sed_pred_bin.float().mean().item()

                    # 2) GT SED (ADPIT 레이블에서 act 축만 꺼내기)
                    # label: [B, T, 6*5*C] 로 flatten 돼 있다고 가정
                    gt = label.view(B, T, 6, 5, C)   # [B, T, 6, 5, C]
                    gt_sed = gt[:, :, :, 0, :] > 0.5 # act 축 [B, T, 6, C]
                    gt_active_ratio = gt_sed.float().mean().item()

                    print(f"[DBG][SED] GT active ratio: {gt_active_ratio:.6f}, "
                        f"Pred active ratio: {pred_active_ratio:.6f}")


            if params['multi_accdoa'] is True:
                sed_np, doa_np, dist_np = decode_tracks(output, params['unique_classes'], params=params)
                sed_np  = sed_np[0]
                doa_np  = doa_np[0]
                dist_np = dist_np[0]

                dist_np = np.nan_to_num(dist_np, nan=0.0, posinf=0.0, neginf=0.0)
                dist_np = np.clip(dist_np, 0.0, 1e4)

                sed_pred0 = sed_np[:, 0, :]
                sed_pred1 = sed_np[:, 1, :]
                sed_pred2 = sed_np[:, 2, :]

                def pack_doa(track_idx):
                    xyz = doa_np[:, track_idx, :, :]
                    return np.concatenate([xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]], axis=-1)

                doa_pred0 = pack_doa(0)
                doa_pred1 = pack_doa(1)
                doa_pred2 = pack_doa(2)
                dist_pred0 = dist_np[:, 0, :]
                dist_pred1 = dist_np[:, 1, :]
                dist_pred2 = dist_np[:, 2, :]

            else:
                sed_logits = output["sed"] if isinstance(output, dict) else output
                doa_out    = output["doa"] if isinstance(output, dict) else None
                B, T, _ = sed_logits.shape
                sed_np = (torch.sigmoid(sed_logits).detach().cpu().numpy() >= thr).astype(np.float32)
                doa_np = doa_out.detach().cpu().numpy().reshape(B, T, params['unique_classes'], 3)
                sed_pred = sed_np[0]
                doa_3d   = doa_np[0]
                doa_pred = np.concatenate([doa_3d[:, :, 0], doa_3d[:, :, 1], doa_3d[:, :, 2]], axis=-1)

            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            if not saw_first:
                print(f"[DBG][VAL] first output file path: {output_file}")
                saw_first = True

            output_dict = {}

            if params['multi_accdoa'] is True:
                T, C = sed_pred0.shape
                for frame_cnt in range(T):
                    for class_cnt in range(C):
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])

                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + C], doa_pred0[frame_cnt][class_cnt + 2 * C], float(dist_pred0[frame_cnt][class_cnt])])
                            if sed_pred1[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + C], doa_pred1[frame_cnt][class_cnt + 2 * C], float(dist_pred1[frame_cnt][class_cnt])])
                            if sed_pred2[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + C], doa_pred2[frame_cnt][class_cnt + 2 * C], float(dist_pred2[frame_cnt][class_cnt])])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + C], doa_pred2[frame_cnt][class_cnt + 2 * C], float(dist_pred2[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2; dist_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + C], doa_fc[class_cnt + 2 * C], float(dist_fc[class_cnt])])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + C], doa_pred0[frame_cnt][class_cnt + 2 * C], float(dist_pred0[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2; dist_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + C], doa_fc[class_cnt + 2 * C], float(dist_fc[class_cnt])])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + C], doa_pred1[frame_cnt][class_cnt + 2 * C], float(dist_pred1[frame_cnt][class_cnt])])
                                doa_fc  = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2; dist_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + C], doa_fc[class_cnt + 2 * C], float(dist_fc[class_cnt])])
                        else:
                            doa_fc  = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3; dist_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_fc[class_cnt], doa_fc[class_cnt + C], doa_fc[class_cnt + 2 * C], float(dist_fc[class_cnt])])
            else:
                T, C = sed_pred.shape
                for frame_cnt in range(T):
                    for class_cnt in range(C):
                        if sed_pred[frame_cnt][class_cnt] > thr: output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt + C], doa_pred[frame_cnt][class_cnt + 2 * C]])

            data_generator.write_output_format_file(output_file, output_dict)
            file_cnt += 1
            test_loss += loss.item()
            nb_test_batches += 1

            if params['quick_test'] and nb_test_batches == 4:
                print("[INFO] Quick test mode: Stopping epoch early after 4 processed batches.")
                break

        if nb_test_batches == 0:
            return float("nan")
        test_loss /= nb_test_batches

    try:
        num_csvs = len(glob.glob(os.path.join(dcase_output_folder, "*.csv")))
        print(f"[DBG][VAL] CSV written: {num_csvs} → {dcase_output_folder}")
    except:
        pass

    return test_loss


# --- Modified train_epoch with Gradient Accumulation & Memory Management ---
def train_epoch(data_generator, optimizer, model, criterion, params, device):
    train_loss = 0.
    model.train()
    
    # [AMP] GradScaler for Mixed Precision
    scaler = GradScaler()

    # --- Batch Counters ---
    processed_batches = 0
    skipped_empty_batches = 0
    skipped_naninf_batches = 0
    total_batches_expected = data_generator.get_total_batches_in_data()
    # --- Batch Counters End ---

    # [중요] Gradient Accumulation 설정
    # 배치 사이즈가 4라면, 8번 모아서 업데이트 -> 실제 배치 사이즈 32 효과
    # 배치 사이즈가 8이라면, 4번 모아서 업데이트 -> 실제 배치 사이즈 32 효과
    accumulation_steps = 8  # 안전하게 8번 모으기 (배치 4 기준)

    loop = tqdm(
        data_generator.generate(),
        total=total_batches_expected,
        desc='Train',
        ncols=100
    )
    
    # 루프 시작 전 Optimizer 초기화
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, values in enumerate(loop):
        batch_filenames = []
        try:
            # --- Data Loading ---
            if params['modality'] == 'audio_visual':
                if len(values) != 4:
                    raise ValueError(f"Expected 4 items from audio-visual generator, got {len(values)}")
                data, vid_feat, target, batch_filenames = values
                data = torch.as_tensor(data, dtype=torch.float32, device=device)
                vid_feat = torch.as_tensor(vid_feat, dtype=torch.float32, device=device)
                target = torch.as_tensor(target, dtype=torch.float32, device=device)
            else: # audio mode
                if len(values) != 3:
                    raise ValueError(f"Expected 3 items from audio generator, got {len(values)}")
                data, target, batch_filenames = values
                data = torch.as_tensor(data, dtype=torch.float32, device=device)
                target = torch.as_tensor(target, dtype=torch.float32, device=device)
                        # --- ACS + Gain augmentation (train 전용) ---
            if params.get('use_acs_gain', False):
                data, target = foa_acs_gain_augment(data, target, params)
            
            # --- [Safety] NaN/Inf Check & Sanitization ---
            if not torch.all(torch.isfinite(data)):
                print(f"\n[WARN][Train] Batch {batch_idx}: Input Data contains NaN/Inf. Sanitizing to 0.")
                data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=-100.0)
            
            if not torch.all(torch.isfinite(target)):
                print(f"\n[WARN][Train] Batch {batch_idx}: Target contains NaN/Inf. Sanitizing to 0.")
                target = torch.nan_to_num(target, nan=0.0)

            # --- Forward Pass (with AMP) ---
            with autocast():
                if params['modality'] == 'audio_visual':
                    output = model(data, vid_feat)
                else:
                    output = model(data)

                loss = criterion(output, target)
                
                # [Accumulation] Loss 나누기
                loss = loss / accumulation_steps

            # Loss값 저장 (로깅용) - 나눈 값이므로 다시 곱해서 원래 스케일로 기록
            loss_item = loss.item() * accumulation_steps

            if not torch.isfinite(loss):
                print(f"\n[ERROR][Train] Batch {batch_idx}: Loss is {loss.item()}! Skipping backward. files={batch_filenames}")
                skipped_naninf_batches += 1
                continue 

            # [Accumulation] Backward만 수행 (메모리 사용량 낮음)
            scaler.scale(loss).backward()
            
            # [Accumulation] Step 수행 (일정 주기마다)
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get('max_grad_norm', 1.0))
                
                scaler.step(optimizer)
                scaler.update()
                
                # 메모리 즉시 해제
                optimizer.zero_grad(set_to_none=True)


            # --- Loss Accumulation ---
            train_loss += loss_item 
            processed_batches += 1 

            if params.get('quick_test', False) and processed_batches >= 4:
                print("[INFO] Quick test mode: Stopping epoch early after 4 processed batches.")
                break

        except Exception as e:
            error_files = batch_filenames[:3] if batch_filenames else ["<unknown>"]
            print(f"\n[FATAL][Train] Error in batch {batch_idx}: {type(e).__name__}: {e}")
            print(f"  Files potentially in batch: {error_files}")
            
            # OOM 발생 시 더 줄이라고 제안하고 종료
            if "OutOfMemoryError" in str(e):
                print("\n[CRITICAL OOM] Even with batch size 8? Try reducing batch_size to 4 in parameters.py!")
                raise e 
            raise 

    if processed_batches > 0:
        train_loss /= processed_batches
    else:
        train_loss = 0.0
        print(f"[WARN][Train Epoch End] No batches were processed in this epoch!")

    print(f"[INFO][Train Epoch End] Processed: {processed_batches}/{total_batches_expected}, "
          f"Skipped (Nan Loss): {skipped_naninf_batches}")

    return train_loss

import torch.backends.cudnn as cudnn

def main(argv):
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cudnn.benchmark = True   # Conv 입력 크기 고정이면 속도↑

    try:
        torch.set_float32_matmul_precision('medium')  # PyTorch 2.x에서 속도↑
    except AttributeError:
        pass
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)
    
    # [중요] 여기서 배치 사이즈를 강제로 4로 낮춤 (OOM 방지 최후의 수단)
    # Gradient Accumulation이 8번 모으므로 실제 배치는 32가 됨.
    print(f"[OOM Prevention] Overwriting batch_size: {params['batch_size']} -> 4")
    params['batch_size'] = 4 

    job_id = 1 if len(argv) < 3 else argv[-1]

    train_splits, val_splits, test_splits = None, None, None

    if params['mode'] == 'dev':
        _ds = params['dataset_dir'].lower()
        if '2020' in _ds:
            test_splits = [1]; val_splits = [2]; train_splits = [[3, 4, 5, 6]]
        elif '2024' in _ds:
            test_splits = [[4]]; val_splits = [[4]]; train_splits = [[1,3]]
        elif '2021' in _ds:
            test_splits = [6]; val_splits = [5]; train_splits = [[1, 2, 3, 4]]
        elif '2022' in _ds:
            test_splits = [[4]]; val_splits = [[4]]; train_splits = [[1, 2, 3]]
        elif '2023' in _ds:
            test_splits = [[4]]; val_splits = [[4]]; train_splits = [[1, 2, 3]]
        else:
            print('ERROR: Unknown dataset splits')
            exit()

        for split_cnt, split in enumerate(test_splits):
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
            print('---------------------------------------------------------------------------------------------------')

            loc_feat = params['dataset']
            if params['dataset'] == 'mic':
                if params['use_salsalite']: loc_feat = '{}_salsa'.format(params['dataset'])
                else: loc_feat = '{}_gcc'.format(params['dataset'])
            loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

            cls_feature_class.create_folder(params['model_dir'])
            unique_name = '{}_{}_{}_split{}_{}_{}'.format(task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat)
            model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
            print("unique_name: {}\n".format(unique_name))

            params['unique_classes'] = 13

            print('Loading training dataset:')
            # [수정] 배치 사이즈가 바뀌었으므로 DataGenerator를 다시 초기화할 때 반영됨
            data_gen_train = cls_data_generator.DataGenerator(params=params, split=train_splits[split_cnt])

            print('Loading validation dataset:')
            data_gen_val = cls_data_generator.DataGenerator(params=params, split=val_splits[split_cnt], shuffle=False, per_file=True)

            tr_files = data_gen_train.get_filelist()
            va_files = data_gen_val.get_filelist()
            print(f"[DBG] train files: {len(tr_files)}  examples: {tr_files[:3]}")
            print(f"[DBG] val   files: {len(va_files)}  examples: {va_files[:3]}")

            if not va_files:
                raise RuntimeError("[VAL] no validation files for split")

            data_in, data_out = data_gen_train.get_data_sizes()
            model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

            if params['finetune_mode']:
                print('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights']))
                if params['pretrained_model_weights'] is None:
                    raise ValueError("Finetuning mode is on, but 'pretrained_model_weights' is not set in parameters.")
                state_dict = torch.load(params['pretrained_model_weights'], map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

            print('---------------- SELD-net (ResNet + Conformer) -------------------')
            print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
            print(model)
            
            dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
            cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
            print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

            score_obj = ComputeSELDResults(params)

            best_val_epoch = -1
            best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999, 999999., 999999.
            patience_cnt = 0

            nb_epoch = params['nb_epochs']
            wd = 0.01 if not params.get('finetune_mode', False) else 0.001

            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'], weight_decay=wd)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            if params['multi_accdoa'] is True:
                criterion = seldnet_model.SeldLoss(params).to(device)
            else:
                criterion = nn.MSELoss().to(device)


            for epoch_cnt in range(nb_epoch):
                start_time = time.time()
                train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device)
                train_time = time.time() - start_time

                start_time = time.time()
                val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
                
                import glob
                _val_csvs = glob.glob(os.path.join(dcase_output_val_folder, "*.csv"))
                if not _val_csvs:
                    raise RuntimeError(f"[VAL] no prediction CSVs written.")

                val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)
                
                scheduler.step(val_seld_scr)
                val_time = time.time() - start_time
                
                if val_seld_scr <= best_seld_scr:
                    best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                    best_rel_dist_err = val_rel_dist_err
                    torch.save(model.state_dict(), model_name)
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                print(
                    'epoch: {}, time: {:0.2f}/{:0.2f}, '
                    'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                    'F/AE/Dist_err/Rel_dist_err/SELD: {}, '
                    'best_val_epoch: {} {}'.format(
                        epoch_cnt, train_time, val_time,
                        train_loss, val_loss,
                        '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr),
                        best_val_epoch,
                        '({:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f})'.format(best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr))
                )
                if patience_cnt > params['patience']:
                    break

            print('Load best model weights')
            model.load_state_dict(torch.load(model_name, map_location='cpu'))

            print('Loading unseen test dataset:')
            data_gen_test = cls_data_generator.DataGenerator(params=params, split=test_splits[split_cnt], shuffle=False, per_file=True)

            dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
            cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
            print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

            test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

            use_jackknife=True
            test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife)

            print('SELD score: {:0.4f}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr))

    if params['mode'] == 'eval':
        print('Loading evaluation dataset:')
        data_gen_eval = cls_data_generator.DataGenerator(params=params, shuffle=False, per_file=True, is_eval=True)

        if params['modality'] == 'audio_visual':
            data_in, vid_data_in, data_out = data_gen_eval.get_data_sizes()
            model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
        else:
            data_in, data_out = data_gen_eval.get_data_sizes()
            model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

        print('Load best model weights')
        model_name = os.path.join(params['model_dir'], '3_1_dev_split0_multiaccdoa_foa_model.h5')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_{}_eval'.format(params['dataset'], loc_output, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise eval results in: {}'.format(dcase_output_test_folder))

        eval_epoch(data_gen_eval, model, dcase_output_test_folder, params, device)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)