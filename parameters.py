# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
# This version is refactored for the ResNet + Conformer model.
# DCASE 2024 Task 3 SOTA-ish configuration, adapted for your setup.

def get_params(argv='1'):
    print("SET: {}".format(argv))
    
    # ########### Default parameters (SOTA-ish baseline) ##############
    params = dict(
        quick_test=False,
        finetune_mode=False,
        
        # [SOTA 1] Feature: SALSA-Lite (set False when using STP-ACC or LogMel-only)
        # ★ 주의: 이 값을 바꾸면 반드시 feature 추출을 다시 해야 함 (extract_features.py)
        use_salsalite=True,  
        
        # SALSA-Lite 피처 추출을 위한 주파수 범위 설정 (Hz)
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,      # 위상 정보가 정확한 저주파 대역 집중
        fmax_spectra_salsalite = 9000,  # 스펙트로그램 상한선
        
        # INPUT PATH (사용자 경로 유지)
        dataset_dir = r'H:\starss23\DCASE_synthetic_dataset_2024',
        feat_label_dir = r'H:\starss23\feature_labels_stpacc',
        model_dir='models',
        dcase_output_dir='results',

        # DATASET LOADING
        mode='dev',
        dataset='foa',
        dist_scale = 1.0,

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        # MODEL TYPE
        modality='audio', 
        multi_accdoa=True, 
        thresh_unify=15,

        # -------- SED Threshold & Post-processing --------
        # DCASE baseline들은 보통 0.3 ~ 0.5 근처를 많이 사용
        sed_threshold=0.30,              # 너무 낮게 두면 (0.05) 거의 전부 1로 켜짐
        sed_median_kernel=5,           
        sed_hysteresis=(0.20, 0.40),   # (on, off) 값 – 필요하면 후처리에서 사용
        sed_prior=0.05,                # SED bias 초기화용 prior (ResNetConformer에서 사용)

        # -------- 손실 가중치 (Balanced) ----------
        # 너무 큰 SED 가중치/pos_weight 때문에 "전부 1로 예측"되는 현상을 완화
        sed_weight  = 4.0,   # 10.0 -> 4.0
        doa_weight  = 1.0,
        dist_weight = 0.2,   # 0.1 -> 0.2 : 위치/거리 쪽도 조금 더 반영

        # -------- SED 로스 설정 ----------
        # 기본: BCE + pos_weight
        use_focal_for_sed = False, 
        focal_gamma       = 2.0,
        sed_pos_weight    = 2.0,  # 10.0 -> 2.0 : positive 과도 페널티 줄이기

        # --- ResNet Parameters ---
        resnet_layers=[2, 2, 2, 2],

        # --- Conformer Parameters (baseline) ---
        conformer_n_layers=6,           # 기본은 6, 실험에서 4~8 정도 사용
        conformer_n_heads=8,
        conformer_conv_kernel_size=31,

        # --- Training Parameters ---
        label_sequence_length=50,
        batch_size=32,       # main()에서 OOM 방지용으로 4로 override 됨
        dropout_rate=0.1,
        nb_epochs=100,
        lr=3e-4,
        patience=50,

        # METRIC
        average='macro',
        segment_based_metrics=False,
        evaluate_distance=True,
        lad_doa_thresh=20,
        lad_dist_thresh=float('inf'),
        lad_reldist_thresh=float('1'),
    )

    # ########### User-defined experiments ##############
    
    if argv == '1':
        # Very small quick test – overfit 여부/파이프라인 체크용
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['batch_size'] = 8
        params['nb_epochs'] = 2


    elif argv == '6':  # SALSA-Lite + Deep Conformer (논문식 셋업)
        print(">>> EXP 6: SALSA-Lite + Deep Conformer (SOTA-style) <<<\n")
        
        # 1. Feature & Dataset
        params['use_salsalite'] = True
        params['feat_label_dir'] = r'H:\starss23\feature_labels_salsalite'  # SALSA-Lite 추출한 폴더로 맞춰줘
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        
        # 2. Model Depth
        params['conformer_n_layers'] = 8
        params['conformer_n_heads']  = 8
        
        # 3. Learning Setup
        params['lr'] = 3e-4
        params['batch_size'] = 8
        
        # 4. Loss Balancing (조금 더 SED에 힘, 하지만 과하지 않게)
        params['sed_weight']     = 4.0
        params['sed_pos_weight'] = 2.0
        params['doa_weight']     = 1.0
        params['dist_weight']    = 0.2
        
        params['use_focal_for_sed'] = False
        
        # 5. Threshold – SALSA-Lite 기준 약간 낮게
        params['sed_threshold'] = 0.25
        
        params['finetune_mode'] = False
        params['pretrained_model_weights'] = None

    elif argv == '7':
        # LogMel + STP-ACC 실험 (기존 FireDOA 계열 세팅과 유사)
        print(">>> EXP 7: LogMel + STP-ACC + IV <<<\n")
        
        # 1. Feature Setup
        params['use_salsalite'] = False
        params['use_stpacc'] = True
        params['feat_label_dir'] = r'H:\starss23\feature_labels_stpacc'
        
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        
        # 2. Model Setup
        params['conformer_n_layers'] = 4
        params['conformer_n_heads']  = 8
        params['batch_size'] = 8
        
        # 3. Learning
        params['lr'] = 1e-4

        # (1) Loss 가중치
        params['sed_weight']  = 4.0
        params['doa_weight']  = 1.0
        params['dist_weight'] = 0.3

        # (2) SED 로스: pos_weight 줄이기
        params['use_focal_for_sed'] = False
        params['sed_pos_weight']    = 2.0

        # (3) SED bias / threshold
        params['sed_prior']     = 0.08
        params['sed_threshold'] = 0.20

    elif argv == '10':
        print(">>> EXP 10: 1st-team style head + ADPIT (simplified loss) <<<\n")

        params['use_salsalite'] = False
        params['use_stpacc']    = True
        params['dataset']       = 'foa'
        params['multi_accdoa']  = True

        params['conformer_n_layers'] = 8
        params['conformer_n_heads']  = 8
        params['batch_size'] = 8

        # LR
        params['lr'] = 3e-4

        # ---- [중요] Loss 밸런스 재조정 ----
        # SED를 너무 세게 밀지 말고, DOA/Dist 쪽 비중을 키워서
        # "방향을 제대로 맞추는" 쪽으로 압박을 줌
        params['sed_weight']     = 2.0      # 4.0 -> 2.0
        params['sed_pos_weight'] = 1.0      # pos_weight 사실상 사용 안 함 (1.0으로 둠)
        params['doa_weight']     = 2.0      # 1.0 -> 2.0
        params['dist_weight']    = 0.7      # 0.3 -> 0.7

        # ---- [SED Loss] focal 안 쓰고, plain BCE만 사용 ----
        params['use_focal_for_sed'] = False
        params['focal_gamma']       = 2.0   # (지금은 사용 안 됨)

        # ---- [중요] Threshold / 후처리 ----
        # 현재 sed_threshold=0.1 이라 Pred active ratio가 0.3까지 튀는 상태라,
        # threshold를 확 올려서 과검출을 강하게 억제
        params['sed_threshold']     = 0.5
        params['sed_median_kernel'] = 5
        params['sed_hysteresis']    = (0.5, 0.7)

        # 거리 스케일
        params['dist_scale'] = 10.0

    else:
        # 다른 argv가 들어오면 기본(default) 세팅 사용
        pass

    # ########### Calculated parameters ##############
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    
    # Path handling
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    # Class Check
    if 'starss23' in params['dataset_dir'].lower():
        params['unique_classes'] = 13
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    else:
        params['unique_classes'] = 13

    print("Parameters:")
    for key, value in params.items():
        print(f"\t{key}: {value}")
    return params
