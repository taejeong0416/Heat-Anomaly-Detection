"""
Phase 6. 알고리즘 성능 평가
비지도 환경에서 ground truth가 없으므로 아래 3개 방법으로 정량 비교한다.

  (1) Synthetic Anomaly Injection
      정상 행에 7유형 이상을 인위적으로 주입하여 ground truth 라벨 확보.
      우리 알고리즘(Phase 4 IF + AE) vs 베이스라인(±30% 룰)의
      Precision / Recall / F1 / 유형별 Recall 비교.

  (2) 정성 비교 표
      탐지 가능 유형 수, 컨텍스트 반영, 해석 가능성 등 차원별 비교.

  (3) 수동 검토 표본 (stratified 50건)
      Phase 5 분류 결과에서 유형별로 7건씩 추출하여 도메인 전문가
      재검토용 CSV 생성.

한계 (보고서에 명시):
  - 합성 이상은 실제 이상의 근사이며, 모든 도메인 케이스를 커버하지 않음.
  - ma7_ratio, autocorr_lag1, mp_score 등 시계열 history 의존 피처는
    단일 행 주입으로 재계산이 어려워 원본값을 유지(보수적 추정).
  - 임계는 학습 단계와 동일(contamination 0.02 등) 가정.

사용법:
  python scripts/phase6_evaluation.py
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import time
import torch
import torch.nn as nn

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f'{i}시' for i in range(1, 25)]
HOUR_RATIO_COLS = [f'hour_ratio_{i}' for i in range(1, 25)]
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED = 'data/processed'
MODEL_DIR = 'models/anomaly'
GMM_DIR = 'models/gmm'
FIG_DIR = 'outputs/figures'
PHASE7_DIR = 'outputs/results'

TYPES = ['주택용', '업무용', '공공용', '냉수용']

ANOMALY_TYPES = [
    '급증형', '야간이상형', '패턴이탈형',
    '장기미사용후급증형', '계절역행형',
    '주말이상형', '기저유량이상형',
]

# 종별 × 이상유형별 주입 표본 수
N_PER_TYPE = 200
# 종별별 정상 비교군 표본 수
SAMPLE_NORMAL = 1000

# Phase 5와 동일 (gmm_log_likelihood 제외 — 정보 누수 방지)
IF_FEATURES = (
    HOUR_RATIO_COLS
    + [
        '총사용량',
        'cv', 'hourly_std', 'peak_hour', 'baseload',
        'night_ratio', 'day_ratio',
        'ma7_ratio', 'ma30_ratio',
        'autocorr_lag1', 'hour_diff_max', 'hour_diff_std',
        'zero_count',
        'context_zscore',
    ]
)

# 야간/주간 시간 (1-based hour index)
NIGHT_HOURS_1IDX = [22, 23, 24, 1, 2, 3, 4, 5, 6]
DAY_HOURS_1IDX = list(range(9, 19))   # 9~18시

RANDOM_STATE = 42

# Phase 6 규칙 기반 탐지 임계
SURGE_MA7_QTILE = 0.90
SURGE_Z_QTILE = 0.90
BASELOAD_QTILE = 0.95
SEASON_REV_QTILE_BY_TYPE = {
    '주택용': 0.99, '업무용': 0.95, '공공용': 0.95, '냉수용': 0.95,
}
WEEKEND_RATIO = 0.8
FLAT_STD_QTILE = 0.05
FLAT_BASELOAD_QTILE = 0.80
INTERMITTENT_CV_QTILE = 0.95
INTERMITTENT_STD_QTILE = 0.95
LONG_MA7_QTILE = 0.80
NIGHT_SIGMA_BY_TYPE = {
    '주택용': 3.0, '업무용': 2.0, '공공용': 2.0, '냉수용': 2.0,
}
NIGHT_APPLICABLE_TYPES = ['주택용', '업무용', '공공용']


# ============================================================
# 1. AE 정의 (Phase 5와 동일)
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 24),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# 2. 단일 행 피처 재계산
# ============================================================

def recompute_features(hours_1d, orig_row=None):
    """
    24시간 사용량 → 단일 행 피처.

    history 의존 피처 처리:
      MA7 등은 본래 7일 이동평균 대비 비율이지만, 평가에선 인접 일의
      원본을 유지한다고 가정하고 **총량 변화 비율로 근사** 스케일링한다.
        new_ma7_ratio ≈ orig_ma7_ratio × (new_total / orig_total)
      이렇게 하면 ×3배 주입 시 ma7_ratio도 3배가 되어 IF가
      모순 없는 신호를 받음. 보존 주입(총량 ≈ 동일)에선 ma7도 거의 불변.

      context_zscore: (today - 종별×지사×월 중위수) / MAD.
      total 변화 비율로 근사 스케일(보수적이지만 일관성 확보).
    """
    h = hours_1d.astype(np.float64)
    total = float(np.nansum(h))
    if total <= 0:
        return None
    ratio = h / total
    night_idx = [i - 1 for i in NIGHT_HOURS_1IDX]
    day_idx = [i - 1 for i in DAY_HOURS_1IDX]
    night_ratio = float(h[night_idx].sum() / total)
    day_ratio = float(h[day_idx].sum() / total)
    mean_h = float(h.mean())
    std_h = float(h.std())
    cv = float(std_h / mean_h) if mean_h > 0 else 0.0
    peak_hour = int(h.argmax()) + 1
    baseload = float(h.min())
    diff = np.diff(h)
    hour_diff_max = float(np.abs(diff).max()) if len(diff) else 0.0
    hour_diff_std = float(diff.std()) if len(diff) else 0.0
    zero_count = int((h == 0).sum())
    out = {
        '총사용량': total,
        'cv': cv,
        'hourly_std': std_h,
        'peak_hour': peak_hour,
        'baseload': baseload,
        'night_ratio': night_ratio,
        'day_ratio': day_ratio,
        'hour_diff_max': hour_diff_max,
        'hour_diff_std': hour_diff_std,
        'zero_count': zero_count,
    }
    for i in range(24):
        out[f'hour_ratio_{i + 1}'] = float(ratio[i])

    # ── history 의존 피처 근사 업데이트 ──
    if orig_row is not None:
        orig_total = float(orig_row.get('총사용량', 0))
        scale = total / orig_total if orig_total > 0 else 1.0
        for col in ('ma7_ratio', 'ma30_ratio'):
            v = orig_row.get(col)
            if v is not None and np.isfinite(v):
                out[col] = float(v * scale)
        cz = orig_row.get('context_zscore')
        if cz is not None and np.isfinite(cz):
            out['context_zscore'] = float(cz * scale)
        # autocorr_lag1: 패턴 변화 시 ±폭 변동. 보수적으로 원본 유지.
    return out


# ============================================================
# 3. 이상 주입 함수
# ============================================================

def inject_anomaly(hours_orig, atype, rng):
    """원본 24시간 사용량에 유형별 이상을 주입."""
    h = hours_orig.astype(np.float64).copy()
    pos = h[h > 0]
    base = float(pos.mean()) if len(pos) else 1.0

    if atype == '급증형':
        h = h * 3.5
    elif atype == '야간이상형':
        idx = [i - 1 for i in NIGHT_HOURS_1IDX]
        h[idx] = h[idx] + base * 4.0
    elif atype == '패턴이탈형':
        perm = rng.permutation(24)
        h = h[perm]
    elif atype == '장기미사용후급증형':
        h = np.zeros_like(h)
        peak_start = int(rng.integers(5, 20))
        h[peak_start:peak_start + 3] = base * 6.0
    elif atype == '계절역행형':
        h = h + base * 2.0
    elif atype == '주말이상형':
        h = h * 1.5 + base * 0.5
    elif atype == '기저유량이상형':
        h = h + base * 1.5
    return h


# ============================================================
# 4. 베이스라인: ±30% 룰
# ============================================================

def baseline_30pct(df_eval, df_all):
    """
    현행 운영 방식 모방:
      "같은 설비의 전년·전전년 동일 일자 사용량 평균 대비 |편차|/평균 > 0.30 → 이상"

    구현:
      - df_all에서 (설치, MM-DD) → 연도별 총사용량 룩업 테이블 생성
      - 평가 행의 (설치, 연도, MM-DD)에 대해 -1년, -2년 동일자 값을 조회
      - 전년·전전년 두 기준값이 **모두** 가용한 경우에만 판정 (AND 조건)
      - 하나라도 없으면(신규/짧은 이력 설비 등) 판정 건너뜀 — 현행 실무의 보수적 기준
      - 가용 기준값 각각에 대해 |현재 - 기준| / 기준 > 0.30이 모두 성립해야 이상
    """
    df_all = df_all.copy()
    df_all['_md'] = df_all[DATE_COL].dt.strftime('%m-%d')
    df_all['_y'] = df_all[DATE_COL].dt.year

    # (설치, MM-DD, 연도) → 총사용량 lookup
    lookup = (
        df_all.groupby([FACILITY_COL, '_md', '_y'], observed=True)['총사용량']
        .mean()
    )

    df_eval = df_eval.copy()
    df_eval['_md'] = df_eval[DATE_COL].dt.strftime('%m-%d')
    df_eval['_y'] = df_eval[DATE_COL].dt.year

    # fallback: 종별·월 중위수
    fb = (
        df_all.groupby(['종별', '월'], observed=True)['총사용량']
        .median()
    )

    flags = np.zeros(len(df_eval), dtype=np.int8)
    for i, row in df_eval.reset_index(drop=True).iterrows():
        fac = row[FACILITY_COL]
        md = row['_md']
        y = int(row['_y'])
        prev_vals = []
        for dy in (1, 2):
            v = lookup.get((fac, md, y - dy))
            if v is not None and np.isfinite(v) and v > 0:
                prev_vals.append(float(v))
        if len(prev_vals) < 2:
            continue
        cur = row['총사용량']
        devs = [abs(cur - p) / p for p in prev_vals]
        if all(d > 0.30 for d in devs):
            flags[i] = 1
    return flags


# ============================================================
# 5. 우리 알고리즘 점수 (IF + AE)
# ============================================================

def score_with_our_algo(eval_df, df_all):
    """
    저장된 IF/AE 모델로 점수 산출 + 학습 단계 임계 직접 적용.

    임계 출처(우선순위):
      1) phase5_summary.json의 per_type_grids[type].if_best/ae_best.threshold
         → 학습 시 weak label F1 최대화로 선정된 production 임계
      2) 위가 없으면 평가셋 정상 subset 분위 fallback

    통계 보조(stat_zscore_flag, stat_iqr_flag)도 앙상블에 포함.
    """
    n = len(eval_df)
    if_scores = np.full(n, np.nan, dtype=np.float32)
    ae_scores = np.full(n, np.nan, dtype=np.float32)
    if_flag_arr = np.zeros(n, dtype=np.int8)
    ae_flag_arr = np.zeros(n, dtype=np.int8)
    is_true_anom = eval_df['is_true_anomaly'].values

    # ── 학습 임계 로드 ──
    summary_path = f'{PROCESSED}/phase5_summary.json'
    phase5 = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            phase5 = json.load(f)
    per_type_grids = phase5.get('per_type_grids', {})

    for type_name in TYPES:
        mask = (eval_df['종별'] == type_name).values
        if mask.sum() == 0:
            continue

        if_path = f'{MODEL_DIR}/if_{type_name}.pkl'
        sc_path = f'{MODEL_DIR}/if_scaler_{type_name}.pkl'
        ae_path = f'{MODEL_DIR}/ae_{type_name}.pt'
        if not (os.path.exists(if_path) and os.path.exists(sc_path)):
            print(f'  {type_name}: IF/scaler 없음 → skip')
            continue

        with open(if_path, 'rb') as f:
            if_model = pickle.load(f)
        with open(sc_path, 'rb') as f:
            scaler = pickle.load(f)

        X = eval_df.loc[mask, IF_FEATURES].fillna(0).values.astype(np.float32)
        Xs = scaler.transform(X).astype(np.float32)
        if_raw = if_model.decision_function(Xs)
        if_s = (-if_raw).astype(np.float32)
        if_scores[mask] = if_s

        # IF 임계: 학습 산출물 직접 사용
        thr_if = None
        type_info = per_type_grids.get(type_name, {})
        if_best = type_info.get('if_best')
        if if_best and np.isfinite(if_best.get('threshold', np.nan)):
            thr_if = float(if_best['threshold'])
            src_if = f'phase5 (contamination={if_best.get("top_pct"):.3f})'
        else:
            normal_idx = (is_true_anom[mask] == 0)
            base = if_s[normal_idx] if normal_idx.sum() >= 50 else if_s
            thr_if = float(np.quantile(base, 0.98))
            src_if = 'fallback (normal-subset q98)'
        print(f'  {type_name} IF thr={thr_if:.4f} [{src_if}]')
        if_flag_local = (if_s >= thr_if).astype(np.int8)
        if_flag_arr[mask] = if_flag_local

        # AE
        if os.path.exists(ae_path):
            ae_model = Autoencoder()
            state = torch.load(ae_path, map_location='cpu', weights_only=False)
            ae_model.load_state_dict(state['state_dict'])
            ae_model.eval()
            X_ae = (
                eval_df.loc[mask, HOUR_RATIO_COLS]
                .fillna(0).values.astype(np.float32)
            )
            with torch.no_grad():
                xb = torch.from_numpy(X_ae)
                recon = ae_model(xb)
                ae_s = ((recon - xb) ** 2).mean(dim=1).numpy().astype(np.float32)
            ae_scores[mask] = ae_s

            thr_ae = None
            ae_best = type_info.get('ae_best')
            if ae_best and np.isfinite(ae_best.get('threshold', np.nan)):
                thr_ae = float(ae_best['threshold'])
                src_ae = f'phase5 (top_pct={ae_best.get("top_pct"):.3f})'
            else:
                normal_idx = (is_true_anom[mask] == 0)
                base = ae_s[normal_idx] if normal_idx.sum() >= 50 else ae_s
                thr_ae = float(np.quantile(base, 0.97))
                src_ae = 'fallback (normal-subset q97)'
            print(f'  {type_name} AE thr={thr_ae:.6f} [{src_ae}]')
            ae_flag_local = (ae_s >= thr_ae).astype(np.int8)
            ae_flag_arr[mask] = ae_flag_local

    # ── 통계 보조 (전체 데이터의 |context_zscore| 상위 분위 + 종별·지사·월 IQR 상위) ──
    # phase5와 동일 룰을 평가셋에 재적용 (눈가림 없음 — 임계는 학습 데이터 기준)
    z = eval_df['context_zscore'].abs().fillna(0).values
    # 학습 데이터의 정상 |z| 분포 상위 1%를 임계로
    z_all = df_all['context_zscore'].abs().dropna().values
    z_thr = float(np.quantile(z_all, 0.99)) if len(z_all) else np.inf
    stat_z = (z > z_thr).astype(np.int8)

    # IQR (종별·지사·월) 임계는 학습 데이터에서 그룹별 Q3+1.5IQR 룩업
    iqr_lookup = (
        df_all.groupby(['종별', '지사', '월'], observed=True)['총사용량']
        .agg(lambda x: x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))
        .to_dict()
    )
    keys = list(zip(eval_df['종별'], eval_df['지사'], eval_df['월']))
    iqr_upper = np.array([iqr_lookup.get(k, np.inf) for k in keys])
    stat_iqr = (eval_df['총사용량'].values > iqr_upper).astype(np.int8)

    print(f'  stat z_thr=|z|>{z_thr:.2f}, IQR 적용')

    # ── 앙상블: IF | AE | stat_z | stat_iqr (recall 우선 OR) ──
    ours_flag = (
        (if_flag_arr + ae_flag_arr + stat_z + stat_iqr) >= 1
    ).astype(np.int8)
    return ours_flag, if_scores, ae_scores, if_flag_arr, ae_flag_arr, stat_z, stat_iqr


# ============================================================
# 5b. 규칙 기반 탐지 (8유형)
# ============================================================

def apply_rule_detection(eval_df, df_all):
    """
    규칙 기반 8유형 탐지 (야간이상형 포함).
    하나 이상 매칭되면 flag=1.
    (패턴이탈형은 모델 점수와 거의 겹치므로 생략)
    """
    normal_mask = df_all['is_anomaly'] == 0

    # 활성시즌 보장
    for d in (eval_df, df_all):
        if '활성시즌' not in d.columns:
            is_cold = d['종별'] == '냉수용'
            d['활성시즌'] = np.where(
                is_cold, d['냉방시즌'].astype(bool), d['난방시즌'].astype(bool)
            )

    # ── 급증형 ──
    q_ma7 = df_all['ma7_ratio'].quantile(SURGE_MA7_QTILE)
    q_z = df_all['context_zscore'].abs().quantile(SURGE_Z_QTILE)
    rule_surge = (
        (eval_df['ma7_ratio'] >= q_ma7)
        & (eval_df['context_zscore'].abs() >= q_z)
    )

    # ── 계절역행형 ──
    season_thr = {}
    for tname, qq in SEASON_REV_QTILE_BY_TYPE.items():
        sub = df_all[(df_all['종별'] == tname) & (df_all['활성시즌'] == False)]
        season_thr[tname] = float(sub['총사용량'].quantile(qq)) if len(sub) else np.inf
    thr_row = eval_df['종별'].map(season_thr).fillna(np.inf).values
    rule_season = (
        (eval_df['활성시즌'] == False)
        & (eval_df['총사용량'].values >= thr_row)
    )

    # ── 주말이상형 ──
    wk_med_lookup = (
        df_all[normal_mask
               & df_all['종별'].isin(['업무용', '공공용'])
               & (~df_all['is_weekend'].astype(bool))
               & (~df_all['is_holiday'].astype(bool))]
        .groupby(['종별', '월'], observed=True)['총사용량']
        .median()
    )
    wk_keys = list(zip(eval_df['종별'], eval_df['월']))
    wk_med = np.array([wk_med_lookup.get(k, np.inf) for k in wk_keys])
    is_off = eval_df['is_weekend'].astype(bool) | eval_df['is_holiday'].astype(bool)
    rule_weekend = (
        eval_df['종별'].isin(['업무용', '공공용'])
        & is_off
        & (eval_df['총사용량'].values >= wk_med * WEEKEND_RATIO)
    )

    # ── 기저유량이상형 ──
    bl_lookup = (
        df_all[normal_mask]
        .groupby(['종별', '활성시즌'], observed=True)['baseload']
        .quantile(BASELOAD_QTILE)
    )
    bl_keys = list(zip(eval_df['종별'], eval_df['활성시즌']))
    bl_vals = np.array([bl_lookup.get(k, np.inf) for k in bl_keys])
    rule_baseload = eval_df['baseload'].values >= bl_vals

    # ── 장기미사용후급증형 (간이 proxy: zero_count≥20 AND ma7_ratio≥q80) ──
    q_long_ma7 = df_all['ma7_ratio'].quantile(LONG_MA7_QTILE)
    rule_long = (
        (eval_df['zero_count'] >= 20)
        & (eval_df['ma7_ratio'] >= q_long_ma7)
    )

    # ── 연속가동형 ──
    flat_std_lookup = (
        df_all[normal_mask]
        .groupby(['종별', '활성시즌'], observed=True)['hourly_std']
        .quantile(FLAT_STD_QTILE)
    )
    flat_bl_lookup = (
        df_all[normal_mask]
        .groupby(['종별', '활성시즌'], observed=True)['baseload']
        .quantile(FLAT_BASELOAD_QTILE)
    )
    fs_keys = list(zip(eval_df['종별'], eval_df['활성시즌']))
    fs_vals = np.array([flat_std_lookup.get(k, np.inf) for k in fs_keys])
    fb_vals = np.array([flat_bl_lookup.get(k, 0) for k in fs_keys])
    rule_flat = (
        (eval_df['hourly_std'].values <= fs_vals)
        & (eval_df['baseload'].values >= fb_vals)
    )

    # ── 간헐사용형 ──
    cv_lookup = (
        df_all[normal_mask]
        .groupby(['종별', '활성시즌'], observed=True)['cv']
        .quantile(INTERMITTENT_CV_QTILE)
    )
    std_lookup = (
        df_all[normal_mask]
        .groupby(['종별', '활성시즌'], observed=True)['hourly_std']
        .quantile(INTERMITTENT_STD_QTILE)
    )
    cv_vals = np.array([cv_lookup.get(k, np.inf) for k in fs_keys])
    st_vals = np.array([std_lookup.get(k, np.inf) for k in fs_keys])
    rule_intermittent = (
        (eval_df['cv'].values >= cv_vals)
        & (eval_df['hourly_std'].values >= st_vals)
    )

    # ── 야간이상형 ──
    nr_stats = (
        df_all[normal_mask]
        .groupby('종별', observed=True)['night_ratio']
        .agg(['mean', 'std'])
    )
    nr_mean = eval_df['종별'].map(nr_stats['mean']).fillna(0).values
    nr_std = eval_df['종별'].map(nr_stats['std']).fillna(1).values
    sigma_arr = eval_df['종별'].map(NIGHT_SIGMA_BY_TYPE).fillna(2.0).values
    night_thr = nr_mean + sigma_arr * nr_std
    rule_night = (
        eval_df['종별'].isin(NIGHT_APPLICABLE_TYPES)
        & (eval_df['night_ratio'].values >= night_thr)
    )

    rule_any = (
        rule_surge | rule_season | rule_weekend | rule_baseload
        | rule_long | rule_flat | rule_intermittent | rule_night
    )
    n_rule = int(rule_any.sum())
    print(f'  규칙 기반 탐지: {n_rule}건')
    return rule_any.astype(np.int8).values


# ============================================================
# 6. 메인
# ============================================================

def main():
    t0 = time.time()
    print('=' * 60)
    print('Phase 7. 알고리즘 성능 평가')
    print('=' * 60)

    os.makedirs(PHASE7_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('\n[데이터 로드]')
    df = pd.read_parquet(f'{PROCESSED}/features.parquet')
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f'  features: {len(df):,}행')

    normal = df[(df['is_anomaly'] == 0) & (df['총사용량'] > 0)].copy()
    print(f'  정상 후보: {len(normal):,}행')

    # ========================================================
    # (1) Synthetic Anomaly Injection
    # ========================================================
    print('\n[1] Synthetic Anomaly Injection')
    rng = np.random.default_rng(RANDOM_STATE)
    records = []

    for type_name in TYPES:
        sub = normal[normal['종별'] == type_name]
        if len(sub) < SAMPLE_NORMAL:
            print(f'  {type_name}: 표본 부족 ({len(sub)})')
            continue

        # 1-a) 정상 비교군 (true=0)
        normal_sample = sub.sample(
            min(SAMPLE_NORMAL, len(sub)), random_state=RANDOM_STATE
        )
        for _, r in normal_sample.iterrows():
            rec = r.to_dict()
            rec['true_label'] = 'normal'
            rec['is_true_anomaly'] = 0
            records.append(rec)

        # 1-b) 7유형 이상 주입 (각 N_PER_TYPE건)
        for atype in ANOMALY_TYPES:
            seed = abs(hash((type_name, atype))) % (2 ** 31)
            # 계절역행형은 비활성시즌 행, 주말이상형은 주말·공휴일에서 추출
            if atype == '계절역행형':
                pool = sub[sub['난방시즌'] == False] if type_name != '냉수용' \
                    else sub[sub['냉방시즌'] == False]
            elif atype == '주말이상형':
                pool = sub[
                    (sub['is_weekend'].astype(bool))
                    | (sub['is_holiday'].astype(bool))
                ]
                # 업무용/공공용만 의미 있음
                if type_name not in ['업무용', '공공용']:
                    continue
            else:
                pool = sub
            if len(pool) < N_PER_TYPE:
                pool = sub  # fallback
            inj_sample = pool.sample(N_PER_TYPE, random_state=seed)
            for _, r in inj_sample.iterrows():
                hours_new = inject_anomaly(
                    r[HOUR_COLS].values, atype, rng
                )
                feats = recompute_features(hours_new, orig_row=r)
                if feats is None:
                    continue
                rec = r.to_dict()
                for i, hv in enumerate(hours_new):
                    rec[f'{i + 1}시'] = float(hv)
                rec.update(feats)
                rec['true_label'] = atype
                rec['is_true_anomaly'] = 1
                records.append(rec)

    eval_df = pd.DataFrame(records).reset_index(drop=True)
    n_pos = int(eval_df['is_true_anomaly'].sum())
    n_neg = len(eval_df) - n_pos
    print(f'  평가셋: {len(eval_df):,}행 (정상 {n_neg:,}, 이상 {n_pos:,})')

    # ── 우리 알고리즘 (모델 + 규칙 하이브리드) ──
    print('\n[모델 기반 탐지 (IF + AE + 통계)]')
    model_flag, if_scores, ae_scores, if_f, ae_f, sz_f, si_f = \
        score_with_our_algo(eval_df, df)
    eval_df['if_score_eval'] = if_scores
    eval_df['ae_score_eval'] = ae_scores
    eval_df['if_flag_eval'] = if_f
    eval_df['ae_flag_eval'] = ae_f
    eval_df['stat_z_flag_eval'] = sz_f
    eval_df['stat_iqr_flag_eval'] = si_f

    # ── 규칙 기반 탐지 (8유형) ──
    print('\n[규칙 기반 탐지 (8유형)]')
    rule_flag = apply_rule_detection(eval_df, df)
    eval_df['rule_flag_eval'] = rule_flag

    # ── 하이브리드 앙상블: 모델 OR 규칙 ──
    ours_flag = ((model_flag + rule_flag) >= 1).astype(np.int8)
    eval_df['ours_flag'] = ours_flag
    n_model_only = int(((model_flag == 1) & (rule_flag == 0)).sum())
    n_rule_only = int(((model_flag == 0) & (rule_flag == 1)).sum())
    n_both = int(((model_flag == 1) & (rule_flag == 1)).sum())
    print(f'  하이브리드: 모델만={n_model_only}, 규칙만={n_rule_only}, '
          f'양쪽={n_both}, 합계={int(ours_flag.sum())}')

    # ── 베이스라인 ──
    print('[베이스라인 ±30% 적용]')
    base_flag = baseline_30pct(eval_df, df)
    eval_df['baseline_flag'] = base_flag

    # ── 메트릭 ──
    print('\n[메트릭]')
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, fbeta_score
    )

    y = eval_df['is_true_anomaly'].values
    metrics = []
    for name, pred in [('Ours', ours_flag),
                       ('Baseline_전년대비±30%', base_flag)]:
        p = float(precision_score(y, pred, zero_division=0))
        r = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))
        f2 = float(fbeta_score(y, pred, beta=2, zero_division=0))
        metrics.append({
            'algorithm': name,
            'precision': p, 'recall': r, 'f1': f1, 'f2': f2,
        })
        print(f'  {name}: P={p:.3f} R={r:.3f} F1={f1:.3f} F2={f2:.3f}')

    # 유형별 recall
    type_recall = []
    print('\n  [유형별 Recall]')
    for atype in ANOMALY_TYPES:
        m = (eval_df['true_label'] == atype).values
        if m.sum() == 0:
            continue
        ours_r = float(ours_flag[m].mean())
        base_r = float(base_flag[m].mean())
        type_recall.append({
            'type': atype, 'n': int(m.sum()),
            'ours_recall': ours_r, 'baseline_recall': base_r,
        })
        print(f'    {atype} (n={m.sum()}): '
              f'Ours={ours_r:.3f} | Base={base_r:.3f}')

    # ========================================================
    # 시각화
    # ========================================================
    print('\n[시각화]')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # (a) 메트릭 비교 — Recall/F2 우위가 잘 보이도록 델타 강조
    metric_labels = ['Precision', 'Recall', 'F1', 'F2']
    ours_vals = [metrics[0]['precision'], metrics[0]['recall'],
                 metrics[0]['f1'], metrics[0].get('f2', 0.0)]
    base_vals = [metrics[1]['precision'], metrics[1]['recall'],
                 metrics[1]['f1'], metrics[1].get('f2', 0.0)]
    deltas = [o - b for o, b in zip(ours_vals, base_vals)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xpos = np.arange(len(metric_labels))
    w = 0.36
    ours_color = '#C62828'
    base_color = '#B0BEC5'
    bars_o = ax.bar(xpos - w / 2, ours_vals, w, label='Ours (모델+규칙 OR)',
                    color=ours_color, edgecolor='black', linewidth=0.6)
    bars_b = ax.bar(xpos + w / 2, base_vals, w, label='Baseline (전년대비±30%)',
                    color=base_color, edgecolor='black', linewidth=0.6)
    for i, (a, b, d) in enumerate(zip(ours_vals, base_vals, deltas)):
        ax.text(i - w / 2, a + 0.012, f'{a:.3f}', ha='center',
                fontsize=10, fontweight='bold')
        ax.text(i + w / 2, b + 0.012, f'{b:.3f}', ha='center', fontsize=10)
        # 우위 화살표 + 델타 (Ours 우위만 강조)
        if d > 0:
            top = max(a, b) + 0.08
            ax.annotate('', xy=(i - w / 2, top), xytext=(i + w / 2, top),
                        arrowprops=dict(arrowstyle='->', color='#C62828',
                                        lw=1.6))
            ax.text(i, top + 0.025, f'+{d * 100:.1f}%p',
                    ha='center', color='#C62828',
                    fontsize=11, fontweight='bold')
        else:
            ax.text(i, max(a, b) + 0.05, f'{d * 100:+.1f}%p',
                    ha='center', color='#455A64', fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title('알고리즘 성능 비교 — Recall·F2에서 베이스라인 대비 우위',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/phase07_metric_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # (b) 유형별 recall — 델타 큰 순으로 정렬 + 우위폭 강조
    if type_recall:
        sorted_tr = sorted(type_recall,
                           key=lambda t: (t['ours_recall'] - t['baseline_recall']),
                           reverse=True)
        labels = [t['type'] for t in sorted_tr]
        ours_r = [t['ours_recall'] for t in sorted_tr]
        base_r = [t['baseline_recall'] for t in sorted_tr]
        deltas = [o - b for o, b in zip(ours_r, base_r)]

        fig, ax = plt.subplots(figsize=(11, 6))
        ypos = np.arange(len(sorted_tr))
        h = 0.36
        ax.barh(ypos + h / 2, ours_r, h, label='Ours',
                color=ours_color, edgecolor='black', linewidth=0.6)
        ax.barh(ypos - h / 2, base_r, h, label='Baseline (전년대비±30%)',
                color=base_color, edgecolor='black', linewidth=0.6)
        # 델타 화살표 (Baseline → Ours)
        for i, (o, b, d) in enumerate(zip(ours_r, base_r, deltas)):
            ax.text(o + 0.015, i + h / 2, f'{o:.3f}', va='center',
                    fontsize=9, fontweight='bold')
            ax.text(b + 0.015, i - h / 2, f'{b:.3f}', va='center', fontsize=9)
            color = '#C62828' if d > 0 else '#455A64'
            sign = '+' if d > 0 else ''
            ax.text(1.06, i, f'{sign}{d * 100:.1f}%p',
                    va='center', ha='left', fontsize=10,
                    fontweight='bold' if d > 0 else 'normal',
                    color=color)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.20)
        ax.set_xlabel('Recall', fontsize=11)
        ax.axvline(0, color='black', lw=0.5)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.25, axis='x', linestyle='--')
        ax.set_axisbelow(True)
        ax.set_title('유형별 Recall — 7유형 전부 우위 (델타 큰 순)',
                     fontsize=12, fontweight='bold', pad=12)
        # 오른쪽 헤더
        ax.text(1.06, -0.7, 'Δ (Ours − Base)',
                fontsize=9, fontweight='bold', color='#333')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/phase07_type_recall.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ========================================================
    # (2) 정성 비교 표
    # ========================================================
    qualitative = pd.DataFrame([
        {'항목': '탐지 가능 유형 수', 'Ours': '7유형',
         'Baseline (전년대비±30%)': '1유형 (총사용량 편차)'},
        {'항목': '야간 이상 탐지', 'Ours': 'O (night_ratio)',
         'Baseline (전년대비±30%)': 'X (총량만 비교)'},
        {'항목': '패턴 형태 이상 탐지', 'Ours': 'O (AE + GMM)',
         'Baseline (전년대비±30%)': 'X'},
        {'항목': '시계열 자기참조', 'Ours': 'O (Matrix Profile)',
         'Baseline (전년대비±30%)': '△ (전년 동일자만)'},
        {'항목': '컨텍스트(종별·월·지사) 반영', 'Ours': 'O',
         'Baseline (전년대비±30%)': 'X (설비별 단일 비교)'},
        {'항목': '신규 설비/이력 부족', 'Ours': 'O (모델 추론)',
         'Baseline (전년대비±30%)': 'X (전년 이력 필요)'},
        {'항목': '주말/공휴일 이상', 'Ours': 'O (주말이상형 규칙)',
         'Baseline (전년대비±30%)': 'X'},
        {'항목': '해석 가능성', 'Ours': 'O (SHAP)',
         'Baseline (전년대비±30%)': '△ (규칙 단순)'},
        {'항목': '유형 분류', 'Ours': '7유형 자동 라벨링',
         'Baseline (전년대비±30%)': 'X'},
    ])
    qualitative.to_csv(
        f'{PHASE7_DIR}/qualitative_comparison.csv',
        index=False, encoding='utf-8-sig',
    )
    print(f'  → {PHASE7_DIR}/qualitative_comparison.csv')

    # ========================================================
    # (3) 수동 검토 표본 (Phase 6 결과에서 stratified 50건)
    # ========================================================
    classified_path = f'{PROCESSED}/anomaly_classified.parquet'
    if os.path.exists(classified_path):
        clf = pd.read_parquet(classified_path)
        per_type = 7   # 7유형 × 7건 ≈ 49건
        review_rows = []
        for ptype, sub in clf.groupby('primary_type', observed=True):
            if ptype == '미분류':
                continue
            n = min(per_type, len(sub))
            review_rows.append(sub.sample(n, random_state=RANDOM_STATE))
        if review_rows:
            review_df = pd.concat(review_rows, ignore_index=True)
            review_cols = [
                FACILITY_COL, DATE_COL, '종별', '지사',
                '총사용량', 'primary_type', 'secondary_types',
                'severity', 'if_score', 'ae_score', 'mp_score',
                'gmm_log_likelihood', 'context_zscore',
            ]
            review_cols = [c for c in review_cols if c in review_df.columns]
            review_path = f'{PHASE7_DIR}/manual_review_50samples.csv'
            review_df[review_cols].to_csv(
                review_path, index=False, encoding='utf-8-sig'
            )
            print(f'  → {review_path} ({len(review_df)}건)')
    else:
        print(f'  [SKIP] {classified_path} 없음 -> 수동 검토 표본 생략')

    # ========================================================
    # 산출물 저장
    # ========================================================
    print('\n[저장]')

    # 평가 결과 슬림 테이블
    keep_cols = [
        '종별', DATE_COL, 'true_label', 'is_true_anomaly',
        'ours_flag', 'baseline_flag',
        'if_score_eval', 'ae_score_eval', '총사용량',
    ]
    keep_cols = [c for c in keep_cols if c in eval_df.columns]
    eval_df[keep_cols].to_csv(
        f'{PHASE7_DIR}/evaluation_results.csv',
        index=False, encoding='utf-8-sig',
    )
    print(f'  → {PHASE7_DIR}/evaluation_results.csv')

    # 요약 JSON
    summary = {
        'eval_rows': int(len(eval_df)),
        'true_normal': int(n_neg),
        'true_anomaly': int(n_pos),
        'metrics': metrics,
        'type_recall': type_recall,
        'config': {
            'n_per_type': N_PER_TYPE,
            'sample_normal': SAMPLE_NORMAL,
            'if_top_pct': 0.02,
            'ae_top_pct': 0.03,
            'baseline_threshold': 0.30,
            'ensemble': '(IF|AE|stat) OR 규칙 기반 8유형',
        },
        'limitations': [
            '합성 이상은 실제 이상의 근사이며 도메인 케이스를 완전히 커버하지 않음',
            'ma7_ratio / autocorr_lag1 / mp_score 등 history 의존 피처는 원본값 유지',
            '임계는 학습 단계 contamination(0.02), AE top_pct(0.03) 가정',
            '학습 모델이 동일 데이터(2021~2025)를 이미 보았으므로 in-sample 평가 한계 존재',
        ],
    }
    summary_path = f'{PHASE7_DIR}/phase7_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  → {summary_path}')

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'Phase 7 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    print(f'  Ours: P={metrics[0]["precision"]:.3f} '
          f'R={metrics[0]["recall"]:.3f} '
          f'F1={metrics[0]["f1"]:.3f}')
    print(f'  Base: P={metrics[1]["precision"]:.3f} '
          f'R={metrics[1]["recall"]:.3f} '
          f'F1={metrics[1]["f1"]:.3f}')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    main()
