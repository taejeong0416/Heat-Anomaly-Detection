"""
Phase 2. 정밀 전처리 실행 스크립트
메모리 효율적 처리를 위해 standalone 스크립트로 실행
결과: data/processed/all_data_clean.parquet, norm_params.json
"""
import pandas as pd
import numpy as np
import json
import shutil
import os
import gc

HOUR_COLS = [f'{i}시' for i in range(1, 25)]
PROCESSED_DIR = 'data/processed'
ARCHIVE_DIR = 'data/processed/archive'
os.makedirs(ARCHIVE_DIR, exist_ok=True)

log = []

def log_step(msg):
    log.append(msg)
    print(f'  {msg}')

print('=' * 60)
print('Phase 2. 정밀 전처리')
print('=' * 60)

# ── 데이터 로드 ──
print('\n[1/8] 데이터 로드...')
df = pd.read_parquet(f'{PROCESSED_DIR}/all_data.parquet')
df['날짜'] = pd.to_datetime(df['날짜'])
initial_rows = len(df)
print(f'  원본: {initial_rows:,}행, {df["설치"].nunique():,}개 설비')

# Archive
dst = f'{ARCHIVE_DIR}/all_data_phase0.parquet'
if not os.path.exists(dst):
    shutil.copy2(f'{PROCESSED_DIR}/all_data.parquet', dst)
    print(f'  Archived -> {dst}')

# 메모리 최적화
for col in HOUR_COLS + ['총사용량']:
    df[col] = df[col].astype('float32')
print(f'  메모리: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')

# ── 2-0. 대상 범위 ──
print('\n[2/8] 대상 범위 설정...')
n = len(df)
df = df[df['종별'] != '냉수용']
log_step(f'냉수용 제외: -{n - len(df):,}행')

# NaN 30%+ 설비
nan_rate = df[HOUR_COLS].isnull().any(axis=1).groupby(df['설치']).mean()
bad_inst = nan_rate[nan_rate >= 0.3].index
n = len(df)
df = df[~df['설치'].isin(bad_inst)]
log_step(f'NaN 30%+ 설비 제외: -{n - len(df):,}행 ({len(bad_inst)}개 설비)')
print(f'  남은: {len(df):,}행, {df["설치"].nunique():,}개 설비')
gc.collect()

# ── 2-1. 음수값 처리 (5단계, 이상유형 계층 순서와 동일) ──
print('\n[3/8] 음수값 처리...')

# Step 1: 검침기리셋 — 시간당 < -10,000 Gcal
mask = (df[HOUR_COLS] < -10000).any(axis=1)
n_reset = mask.sum()
if n_reset > 0:
    df = df[~mask]
log_step(f'[검침기리셋] 행 제거: -{n_reset:,}행')

# Step 2: 총량음수 — 총사용량 < 0
mask = df['총사용량'] < 0
n_neg_total = mask.sum()
if n_neg_total > 0:
    df = df[~mask]
log_step(f'[총량음수] 행 제거: -{n_neg_total:,}행')

# Step 3: 다중음수이상 — 하루 음수 3개+
neg_count_per_row = (df[HOUR_COLS] < 0).sum(axis=1)
mask_3plus = neg_count_per_row >= 3
n_3plus = mask_3plus.sum()
if n_3plus > 0:
    df = df[~mask_3plus]
log_step(f'[다중음수이상] 행 제거: -{n_3plus:,}행')

# Step 4: 잔여 경미한 음수 (-1 < x < 0) → 0 클리핑 (계측 오차)
clip_cells = 0
for col in HOUR_COLS:
    m = (df[col] < 0) & (df[col] > -1)
    clip_cells += m.sum()
    if m.any():
        df.loc[m, col] = 0
log_step(f'경미한 음수(-1<x<0) 클리핑: {clip_cells:,}개 셀')

# Step 5: 잔여 큰 음수 (≤ -1, 1~2개) → 결측 변환 (이후 보간에서 처리)
nan_convert_cells = 0
for col in HOUR_COLS:
    m = df[col] <= -1
    nan_convert_cells += m.sum()
    if m.any():
        df.loc[m, col] = np.nan
log_step(f'큰 음수(≤-1) 결측 변환: {nan_convert_cells:,}개 셀')
gc.collect()

# ── 2-2. 극단값 처리 ──
print('\n[4/8] 극단값 처리...')

thresholds = {}
for t in ['주택용', '업무용', '공공용']:
    mask = df['종별'] == t
    # 시간당 값 통합 99.9%ile (메모리 절약: 컬럼별 계산 후 통합)
    all_vals = []
    for col in HOUR_COLS:
        v = df.loc[mask, col].dropna().values
        all_vals.append(v)
    all_vals = np.concatenate(all_vals)
    q999 = float(np.percentile(all_vals, 99.9))
    thresholds[t] = round(q999, 4)
    print(f'  {t}: 99.9%ile = {q999:.4f} Gcal')
    del all_vals
gc.collect()

# 극단값 행 제거 (threshold x 100)
remove_mask = pd.Series(False, index=df.index)
for t in ['주택용', '업무용', '공공용']:
    mask = df['종별'] == t
    th = thresholds[t]
    for col in HOUR_COLS:
        extreme = mask & (df[col] >= th * 100)
        remove_mask = remove_mask | extreme

n_extreme = remove_mask.sum()
if n_extreme > 0:
    types_removed = df.loc[remove_mask, '종별'].value_counts().to_dict()
    df = df[~remove_mask]
    print(f'  극단값 행 제거: {n_extreme:,}건 {types_removed}')
log_step(f'극단값 행 제거: -{n_extreme:,}행')

# Clip
for t in ['주택용', '업무용', '공공용']:
    mask = df['종별'] == t
    th = thresholds[t]
    n_clipped = 0
    for col in HOUR_COLS:
        m = mask & (df[col] > th)
        n_clipped += m.sum()
        if m.any():
            df.loc[m, col] = th
    print(f'  {t}: {n_clipped:,}개 셀 clip -> {th}')
log_step(f'극단값 clip 완료')
gc.collect()

# ── 2-3. 결측치 처리 ──
print('\n[5/8] 결측치 처리...')

nan_count = df[HOUR_COLS].isnull().sum(axis=1)

# NaN=24
n = len(df)
mask24 = nan_count == 24
n_nan24 = mask24.sum()

# NaN=24 분포 기록
nan24_monthly = df.loc[mask24, '월'].value_counts().sort_index().to_dict() if n_nan24 > 0 else {}
nan24_branch = df.loc[mask24, '지사'].value_counts().head(5).to_dict() if n_nan24 > 0 else {}
print(f'  NaN=24: {n_nan24:,}건')
print(f'    월별: {nan24_monthly}')
print(f'    지사 상위5: {nan24_branch}')

df = df[~mask24]
log_step(f'NaN=24 행 제거: -{n_nan24:,}행')

# NaN 3~23
nan_count = df[HOUR_COLS].isnull().sum(axis=1)
mask_partial = (nan_count >= 3) & (nan_count <= 23)
n_partial = mask_partial.sum()
df = df[~mask_partial]
log_step(f'NaN 3~23 행 제거: -{n_partial:,}행')

# NaN 1~2 보간
nan_count = df[HOUR_COLS].isnull().sum(axis=1)
interp_mask = (nan_count >= 1) & (nan_count <= 2)
n_interp = interp_mask.sum()
print(f'  NaN 1~2 보간 대상: {n_interp:,}행')

# 보간 상세 정보 저장: NaN 개수 + 최대 연속 NaN 길이
df['보간_개수'] = 0
df.loc[interp_mask, '보간_개수'] = nan_count[interp_mask].astype(int)

def _max_consecutive_nan_batch(hour_data):
    """행별 최대 연속 NaN 길이 계산 (벡터화)"""
    is_nan = hour_data.isnull().values  # shape: (n_rows, 24)
    max_runs = np.zeros(len(is_nan), dtype=int)
    current = np.zeros(len(is_nan), dtype=int)
    for col_idx in range(24):
        mask_col = is_nan[:, col_idx]
        current = np.where(mask_col, current + 1, 0)
        max_runs = np.maximum(max_runs, current)
    return max_runs

df['보간_최대연속'] = 0
if n_interp > 0:
    idx = df.index[interp_mask]
    df.loc[idx, '보간_최대연속'] = _max_consecutive_nan_batch(df.loc[idx, HOUR_COLS])

    # 연속 NaN 분포 출력
    consec_dist = df.loc[idx, '보간_최대연속'].value_counts().sort_index()
    print(f'  최대 연속 NaN 분포:')
    for k, v in consec_dist.items():
        print(f'    연속 {k}개: {v:,}건 ({v/n_interp*100:.1f}%)')

    # 보간 (행 단위 선형)
    df.loc[idx, HOUR_COLS] = df.loc[idx, HOUR_COLS].interpolate(
        axis=1, method='linear', limit_direction='both'
    )
    # 보간 후 음수 clip
    for col in HOUR_COLS:
        m = df[col] < 0
        if m.any():
            df.loc[m, col] = 0

log_step(f'NaN 1~2 보간: {n_interp:,}행')
remaining_nan = df[HOUR_COLS].isnull().sum().sum()
print(f'  NaN 잔존: {remaining_nan}건')
gc.collect()

# ── 2-4. 총사용량 재계산 ──
print('\n[6/8] 총사용량 재계산...')
df['총사용량'] = df[HOUR_COLS].sum(axis=1)
print(f'  완료')

# ── 2-5. 정규화 파라미터 ──
print('\n[7/8] 정규화 파라미터 산출...')
norm_params = {}
cols = HOUR_COLS + ['총사용량']
for t in ['주택용', '업무용', '공공용']:
    for season in [True, False]:
        mask = (df['종별'] == t) & (df['난방시즌'] == season)
        s_label = '난방' if season else '비난방'
        key = f'{t}_{s_label}'
        means = df.loc[mask, cols].mean()
        stds = df.loc[mask, cols].std().replace(0, 1)
        norm_params[key] = {
            'mean': {k: round(float(v), 6) for k, v in means.items()},
            'std': {k: round(float(v), 6) for k, v in stds.items()},
            'n_rows': int(mask.sum())
        }
        print(f'  {key}: N={mask.sum():>9,}, mean={float(means["총사용량"]):.4f}, std={float(stds["총사용량"]):.4f}')

with open(f'{PROCESSED_DIR}/norm_params.json', 'w', encoding='utf-8') as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

# ── 2-6. 품질 검증 ──
print('\n[8/8] 품질 검증...')
checks = {}
checks['neg_hourly'] = int((df[HOUR_COLS] < 0).sum().sum())
checks['nan_hourly'] = int(df[HOUR_COLS].isnull().sum().sum())
checks['neg_total'] = int((df['총사용량'] < 0).sum())
checks['interp_rate'] = round((df['보간_개수'] > 0).sum() / len(df) * 100, 3)

original_counts = {'주택용': 8399121, '업무용': 6468701, '공공용': 1788904}
checks['removal_rates'] = {}
for t in ['주택용', '업무용', '공공용']:
    current = int((df['종별'] == t).sum())
    rate = round((1 - current / original_counts[t]) * 100, 2)
    checks['removal_rates'][t] = {'original': original_counts[t], 'current': current, 'rate': rate}

for k, v in checks.items():
    if k == 'removal_rates':
        for t, d in v.items():
            status = 'PASS' if d['rate'] < 3 else 'FAIL'
            print(f'  [{status}] {t} 제거율: {d["rate"]}%')
    else:
        if 'rate' in k:
            status = 'PASS' if v < 0.6 else 'WARN'
        else:
            status = 'PASS' if v == 0 else 'FAIL'
        print(f'  [{status}] {k}: {v}')

# ── 저장 ──
print('\n저장 중...')
output_path = f'{PROCESSED_DIR}/all_data_clean.parquet'
df.to_parquet(output_path, index=False)
file_size = os.path.getsize(output_path) / (1024**2)

# 처리 요약도 JSON으로 저장 (노트북 시각화용)
summary = {
    'initial_rows': initial_rows,
    'final_rows': len(df),
    'preservation_rate': round(len(df) / initial_rows * 100, 1),
    'thresholds': thresholds,
    'log': log,
    'checks': checks,
    'nan24_monthly': {str(k): int(v) for k, v in nan24_monthly.items()},
    'nan24_branch': {str(k): int(v) for k, v in nan24_branch.items()},
}
with open(f'{PROCESSED_DIR}/phase2_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f'\n{"=" * 60}')
print(f'전처리 완료!')
print(f'{"=" * 60}')
for entry in log:
    print(f'  {entry}')
print(f'\n  원본: {initial_rows:,}행 -> 최종: {len(df):,}행 ({summary["preservation_rate"]}%)')
print(f'  파일: {output_path} ({file_size:.1f} MB)')
print(f'  파라미터: {PROCESSED_DIR}/norm_params.json')
print(f'  요약: {PROCESSED_DIR}/phase2_summary.json')
