"""
Phase 2. Feature Engineering
이상 탐지 모델을 위한 피처 엔지니어링

학술 근거:
  Capozzoli et al. (2018), Mathieu et al. (2011) — 기본 통계
  Tureczek et al. (2018) — 시간대 비율 (비식별화 데이터에 유리)
  Janetzko et al. (2014) — 이동평균 대비 비율
  Box et al. (2015) — 자기상관
  Aggarwal (2017) — 시간 변화율
  PNNL (2015) — zero consumption

사용법:
  python scripts/phase2_feature.py
"""

import pandas as pd
import numpy as np
import json
import os
import gc
import time

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f'{i}시' for i in range(1, 25)]
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED_DIR = 'data/processed'
FIG_DIR = 'outputs/figures'

# 한국 공휴일 (2021-2025) — 데이터에 공휴일 컬럼 없으므로 직접 정의
KOREA_HOLIDAYS = pd.to_datetime([
    # 2021
    '2021-01-01', '2021-02-11', '2021-02-12', '2021-03-01',
    '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15',
    '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03',
    '2021-10-09', '2021-12-25',
    # 2022
    '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02',
    '2022-03-01', '2022-03-09', '2022-05-05', '2022-05-08',
    '2022-06-01', '2022-06-06', '2022-08-15', '2022-09-09',
    '2022-09-10', '2022-09-12', '2022-10-03', '2022-10-09',
    '2022-10-10', '2022-12-25',
    # 2023
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23',
    '2023-01-24', '2023-03-01', '2023-05-05', '2023-05-27',
    '2023-05-29', '2023-06-06', '2023-08-15', '2023-09-28',
    '2023-09-29', '2023-09-30', '2023-10-02', '2023-10-03',
    '2023-10-09', '2023-12-25',
    # 2024
    '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11',
    '2024-02-12', '2024-03-01', '2024-04-10', '2024-05-05',
    '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15',
    '2024-09-16', '2024-09-17', '2024-09-18', '2024-10-01',
    '2024-10-03', '2024-10-09', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30',
    '2025-03-01', '2025-03-03', '2025-05-05', '2025-05-06',
    '2025-06-06', '2025-08-15', '2025-10-03', '2025-10-05',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09',
    '2025-12-25'
]).normalize()

# 로그
log = []


def log_step(msg):
    log.append(msg)
    print(f'  {msg}')


# ============================================================
# 1. 피처 엔지니어링
# ============================================================

def engineer_features(df):
    """
    24시간 시계열 → 이상 탐지용 피처 생성 (벡터화)

    생성 피처 (13개 신규):
      기본통계: cv, hourly_std, peak_hour, baseload
      패턴비율: hour_ratio_1~24, night_ratio, day_ratio
      추세비교: ma7_ratio, ma30_ratio
      시계열:   autocorr_lag1, hour_diff_max, hour_diff_std, zero_count

    기존 컬럼 활용 (재계산 안 함):
      총사용량, 난방시즌, 냉방시즌, 계절, 월, 요일, 종별, 지사

    신규 메타 (2개):
      is_holiday, is_weekend
    """
    hour_data = df[HOUR_COLS].values.astype(np.float32)
    n_rows = len(hour_data)

    print(f'[피처 생성 시작] {n_rows:,}행 처리\n')

    # ─────────────────────────────────────
    # [1/5] 기본 통계
    # ─────────────────────────────────────
    print('  [1/5] 기본 통계 피처...')

    daily_mean = np.nanmean(hour_data, axis=1)
    daily_std = np.nanstd(hour_data, axis=1)
    df['hourly_std'] = daily_std
    df['cv'] = np.where(daily_mean > 0, daily_std / daily_mean, np.nan)

    # 피크 시간 (벡터화)
    all_nan_mask = np.all(np.isnan(hour_data), axis=1)
    safe_data = np.where(np.isnan(hour_data), -np.inf, hour_data)
    peak_hour = np.argmax(safe_data, axis=1) + 1
    peak_hour = np.where(all_nan_mask, np.nan, peak_hour)
    df['peak_hour'] = peak_hour
    del safe_data
    gc.collect()

    df['baseload'] = np.nanmin(hour_data, axis=1)

    log_step(f'기본 통계 완료: hourly_std, cv, peak_hour, baseload')

    # ─────────────────────────────────────
    # [2/5] 패턴 비율
    # ─────────────────────────────────────
    print('  [2/5] 패턴 비율 피처...')

    daily_sum = np.nansum(hour_data, axis=1)
    daily_sum_safe = np.where(daily_sum > 0, daily_sum, np.nan)

    # 시간대별 비율 — 컬럼별 할당 (pd.concat 대신)
    for h in range(24):
        df[f'hour_ratio_{h+1}'] = hour_data[:, h] / daily_sum_safe

    # 야간 비율 (22-06시: 인덱스 21,22,23,0,1,2,3,4,5)
    night_idx = [21, 22, 23, 0, 1, 2, 3, 4, 5]
    night_sum = np.nansum(hour_data[:, night_idx], axis=1)
    df['night_ratio'] = night_sum / daily_sum_safe

    # 주간 비율 (09-18시: 인덱스 8~17)
    day_idx = list(range(8, 18))
    day_sum = np.nansum(hour_data[:, day_idx], axis=1)
    df['day_ratio'] = day_sum / daily_sum_safe

    log_step(f'패턴 비율 완료: hour_ratio_1~24, night_ratio, day_ratio')

    # ─────────────────────────────────────
    # [3/5] 추세 비교
    # ─────────────────────────────────────
    print('  [3/5] 추세 비교 피처...')

    df.sort_values([FACILITY_COL, DATE_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 총사용량 기반 이동평균 (daily_mean = 총사용량/24이므로 비율은 동일)
    ma7 = df.groupby(FACILITY_COL, sort=False)['총사용량'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['ma7_ratio'] = np.where(ma7 > 0, df['총사용량'] / ma7, np.nan)

    ma30 = df.groupby(FACILITY_COL, sort=False)['총사용량'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    df['ma30_ratio'] = np.where(ma30 > 0, df['총사용량'] / ma30, np.nan)
    del ma7, ma30
    gc.collect()

    log_step(f'추세 비교 완료: ma7_ratio, ma30_ratio')

    # ─────────────────────────────────────
    # [4/5] 시계열 패턴
    # ─────────────────────────────────────
    print('  [4/5] 시계열 패턴 피처...')

    # Lag-1 자기상관 (컬럼별 누적 — 메모리 절약)
    n_rows = len(hour_data)
    ac_n = np.zeros(n_rows, dtype=np.float32)
    ac_sx = np.zeros(n_rows, dtype=np.float32)
    ac_sy = np.zeros(n_rows, dtype=np.float32)
    ac_sxy = np.zeros(n_rows, dtype=np.float32)
    ac_sx2 = np.zeros(n_rows, dtype=np.float32)
    ac_sy2 = np.zeros(n_rows, dtype=np.float32)

    # 시간 변화율도 동시에 컬럼별 계산
    hd_max = np.full(n_rows, -np.inf, dtype=np.float32)
    hd_sum = np.zeros(n_rows, dtype=np.float64)
    hd_sum2 = np.zeros(n_rows, dtype=np.float64)
    hd_n = np.zeros(n_rows, dtype=np.int32)

    for i in range(23):
        xi = hour_data[:, i]
        yi = hour_data[:, i + 1]
        v = ~(np.isnan(xi) | np.isnan(yi))
        xs = np.where(v, xi, 0).astype(np.float32)
        ys = np.where(v, yi, 0).astype(np.float32)
        ac_n += v
        ac_sx += xs
        ac_sy += ys
        ac_sxy += xs * ys
        ac_sx2 += xs * xs
        ac_sy2 += ys * ys

        # hour diff
        d = yi - xi
        vd = ~np.isnan(d)
        ad = np.abs(d)
        hd_max = np.where(vd & (ad > hd_max), ad, hd_max)
        hd_sum += np.where(vd, d, 0)
        hd_sum2 += np.where(vd, d * d, 0)
        hd_n += vd.astype(np.int32)

    # autocorrelation: r = (n*Sxy - Sx*Sy) / sqrt((n*Sx2 - Sx^2)*(n*Sy2 - Sy^2))
    num = ac_n * ac_sxy - ac_sx * ac_sy
    den = np.sqrt((ac_n * ac_sx2 - ac_sx**2) * (ac_n * ac_sy2 - ac_sy**2))
    df['autocorr_lag1'] = np.where((den > 0) & (ac_n >= 3), num / den, np.nan)
    del ac_n, ac_sx, ac_sy, ac_sxy, ac_sx2, ac_sy2, num, den

    # hour_diff_max, hour_diff_std
    hd_mean = np.where(hd_n > 0, hd_sum / hd_n, np.nan)
    hd_var = np.where(hd_n > 0, hd_sum2 / hd_n - hd_mean**2, np.nan)
    df['hour_diff_max'] = np.where(hd_max == -np.inf, np.nan, hd_max)
    df['hour_diff_std'] = np.sqrt(np.maximum(hd_var, 0))
    del hd_max, hd_sum, hd_sum2, hd_n, hd_mean, hd_var
    gc.collect()

    # 0의 개수
    df['zero_count'] = np.sum(hour_data == 0, axis=1)

    log_step(f'시계열 패턴 완료: autocorr_lag1, hour_diff_max, hour_diff_std, zero_count')

    # ─────────────────────────────────────
    # [5/5] 메타 정보 (기존 컬럼 활용 + 신규 2개)
    # ─────────────────────────────────────
    print('  [5/5] 메타 정보 피처...')

    # 기존 컬럼 확인 (난방시즌, 냉방시즌, 계절, 월, 요일, 종별, 지사)
    existing = ['난방시즌', '냉방시즌', '계절', '월', '요일', '종별', '지사']
    for col in existing:
        if col not in df.columns:
            print(f'    경고: {col} 컬럼 없음')

    # 공휴일 (데이터에 없으므로 생성)
    date_normalized = df[DATE_COL].dt.normalize()
    df['is_holiday'] = date_normalized.isin(KOREA_HOLIDAYS).astype(np.int8)

    # 주말 (요일에서 파생 가능하지만 거리 기반 모델용 이진 플래그)
    df['is_weekend'] = (df[DATE_COL].dt.dayofweek >= 5).astype(np.int8)

    log_step(f'메타 정보 완료: is_holiday, is_weekend (기존 컬럼 7개 유지)')

    del hour_data
    gc.collect()

    # ─────────────────────────────────────
    # NaN → 0 채우기 (총사용량=0인 날 발생분)
    # ─────────────────────────────────────
    nan_fill_cols = (
        ['cv', 'night_ratio', 'day_ratio', 'ma7_ratio', 'ma30_ratio',
         'autocorr_lag1', 'hour_diff_max', 'hour_diff_std']
        + [f'hour_ratio_{h}' for h in range(1, 25)]
    )
    for col in nan_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    log_step(f'NaN→0 채움: {len(nan_fill_cols)}개 피처 (총사용량=0 행)')

    print(f'\n[피처 생성 완료] 총 {len(df.columns)}개 컬럼')
    return df


# ============================================================
# 메인 파이프라인
# ============================================================

def run_feature_engineering():
    """피처 생성 실행"""
    t_start = time.time()

    print('=' * 60)
    print('Phase 3. Feature Engineering')
    print('=' * 60)

    os.makedirs(FIG_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    input_path = f'{PROCESSED_DIR}/all_data_clean.parquet'

    import pyarrow.parquet as pq
    _table = pq.read_table(input_path)
    df = _table.to_pandas(self_destruct=True)
    del _table
    gc.collect()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f'  입력: {len(df):,}행, {df[FACILITY_COL].nunique():,}개 설비')
    print(f'  기간: {df[DATE_COL].min().date()} ~ {df[DATE_COL].max().date()}')
    print(f'  컬럼: {len(df.columns)}개')

    # ── 피처 생성 ──
    df = engineer_features(df)
    gc.collect()

    # ── 피처 요약 ──
    new_features = [
        'cv', 'hourly_std', 'peak_hour', 'baseload',
        'night_ratio', 'day_ratio',
        'ma7_ratio', 'ma30_ratio',
        'autocorr_lag1', 'hour_diff_max', 'hour_diff_std',
        'zero_count', 'is_holiday', 'is_weekend'
    ]
    hour_ratio_features = [f'hour_ratio_{h}' for h in range(1, 25)]

    print(f'\n[피처 요약]')
    print(f'  신규 피처: {len(new_features)}개')
    print(f'  시간대 비율: {len(hour_ratio_features)}개')
    print(f'  기존 컬럼: 총사용량, 난방시즌, 냉방시즌, 계절, 월, 요일, 종별, 지사')
    print(f'  전체 컬럼: {len(df.columns)}개')

    # ── 품질 검증 ──
    print(f'\n[품질 검증]')
    for feat in new_features:
        if feat in df.columns:
            n_nan = df[feat].isna().sum()
            n_inf = np.isinf(df[feat].values).sum() if df[feat].dtype in ['float32', 'float64'] else 0
            pct_nan = n_nan / len(df) * 100
            status = 'PASS' if pct_nan < 5 else 'WARN'
            print(f'  [{status}] {feat}: NaN={n_nan:,} ({pct_nan:.2f}%), Inf={n_inf:,}')

    # ── 저장 ──
    print('\n저장 중...')
    output_path = f'{PROCESSED_DIR}/features.parquet'
    df.to_parquet(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 ** 2)

    summary = {
        'input_rows': len(df),
        'input_facilities': int(df[FACILITY_COL].nunique()),
        'total_columns': len(df.columns),
        'new_features': new_features,
        'hour_ratio_features': hour_ratio_features,
        'existing_features_used': [
            '총사용량', '난방시즌', '냉방시즌', '계절', '월', '요일', '종별', '지사'
        ],
        'log': log,
    }
    with open(f'{PROCESSED_DIR}/phase3_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 최종 요약 ──
    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'피처 생성 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    for entry in log:
        print(f'  {entry}')
    print(f'\n  파일: {output_path} ({file_size:.1f} MB)')
    print(f'  요약: {PROCESSED_DIR}/phase3_summary.json')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    run_feature_engineering()
