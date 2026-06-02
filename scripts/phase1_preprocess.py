"""
Phase 1. 정밀 전처리 파이프라인 (통합)
PNNL-24331 (2015) + Quesada et al. (2024) + 데이터 분포 분석 기반

9단계 전처리 + 근거 분석 시각화 코드 통합

사용법:
  python scripts/phase1_preprocess.py              # 전처리 실행
  python scripts/phase1_preprocess.py --analyze     # 근거 분석만 실행 (시각화)
  python scripts/phase1_preprocess.py --all         # 근거 분석 + 전처리 모두
"""

import pandas as pd
import numpy as np
import json
import shutil
import os
import gc
import time
import argparse

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f'{i}시' for i in range(1, 25)]
ALL_TYPES = ['주택용', '업무용', '공공용', '냉수용']
HEATING_TYPES = ['주택용', '업무용', '공공용']
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED_DIR = 'data/processed'
ARCHIVE_DIR = 'data/processed/archive'
FIG_DIR = 'outputs/figures'

# 임계치 (각 단계의 근거 분석 결과 반영)
OFFSET_WINDOW = 2              # 2단계: 인접 양수 탐색 윈도우 (PNNL Section 2.4.1)
DAILY_NAN_THRESHOLD = 3        # 4단계: 분포 분석 elbow at 3 (PNNL Section 2.3)
CONSEC_NAN_THRESHOLD = 2       # 5단계: PNNL Section 3 Option 2 (1시간 초과)
MIN_OBSERVATION_DAYS = 365     # 6단계: Quesada et al. 2024
MAX_MISSING_RATIO = 0.28       # 7단계: 데이터 분포 단절점 분석
MAX_PROCESSED_RATIO = 0.30     # 8단계: 데이터 분포 자연스러운 분기점
EXTREME_PERCENTILE = 99.9      # 9단계: 종별 시간당 사용량 상한
EXTREME_MULTIPLIER = 100       # 9단계: threshold × 100 이상이면 행 제거

# 로그
log = []


def log_step(msg):
    log.append(msg)
    print(f'  {msg}')


# ============================================================
# Phase 1: 음수 처리 (PNNL Section 2.4)
# ============================================================

def step1_remove_daily_negative_total(df):
    """
    1단계: 일별 총사용량 음수 행 제거
    근거: PNNL Table 2-2 — 일별 합계가 음수면 검침기 리셋 또는 심각한 오류
    """
    daily_sum = df[HOUR_COLS].sum(axis=1, skipna=True)
    mask = daily_sum < 0
    n_removed = mask.sum()
    df = df[~mask].reset_index(drop=True)

    log_step(f'[1단계] 일별 총사용량 음수 행 제거: -{n_removed:,}행 (PNNL Table 2-2)')
    return df


def step2_offset_negative_processing(df):
    """
    2단계: 인접 양수 상쇄 음수 처리 (PNNL Section 2.4.1)
    각 음수에 대해 ±2시간 윈도우 내 가장 큰 인접 양수 탐색:
      Case 1 (합 ≤ 0): 음수 + 양수 모두 NaN, 사이의 0값도 NaN
      Case 2 (합 > 0, 영향구간 ≤ 1시간): 합을 분배 보간
      Case 3 (합 > 0, 영향구간 > 1시간): NaN (자동 운영 보수 정책)
    """
    data_array = df[HOUR_COLS].values.astype(np.float32)

    has_negative = (data_array < 0).any(axis=1)
    rows_with_neg = np.where(has_negative)[0]

    counters = {'case1': 0, 'case2': 0, 'case3': 0}

    for row_idx in rows_with_neg:
        values = data_array[row_idx].copy()
        flags = ['collected'] * len(values)

        neg_positions = np.where(~np.isnan(values) & (values < 0))[0]

        for neg_idx in neg_positions:
            if flags[neg_idx] != 'collected':
                continue

            neg_value = values[neg_idx]
            start = max(0, neg_idx - OFFSET_WINDOW)
            end = min(len(values), neg_idx + OFFSET_WINDOW + 1)

            candidates = []
            for i in range(start, end):
                if i == neg_idx:
                    continue
                if (not np.isnan(values[i])
                        and values[i] > 0
                        and flags[i] == 'collected'):
                    candidates.append((i, values[i]))

            if len(candidates) == 0:
                continue  # 3단계로 위임

            pos_idx, pos_value = max(candidates, key=lambda x: x[1])
            sum_value = neg_value + pos_value
            duration = abs(pos_idx - neg_idx)

            if sum_value <= 0:
                # Case 1: 합 ≤ 0 → 양쪽 모두 NaN
                values[neg_idx] = np.nan
                values[pos_idx] = np.nan
                flags[neg_idx] = 'nan_offset'
                flags[pos_idx] = 'nan_offset'
                lo, hi = min(neg_idx, pos_idx), max(neg_idx, pos_idx)
                for i in range(lo + 1, hi):
                    if values[i] == 0:
                        values[i] = np.nan
                        flags[i] = 'nan_offset_zero'
                counters['case1'] += 2

            elif sum_value > 0 and duration <= 1:
                # Case 2: 합 > 0, 인접 → 분배 보간
                avg_value = sum_value / (duration + 1)
                values[neg_idx] = avg_value
                values[pos_idx] = avg_value
                flags[neg_idx] = 'interpolated'
                flags[pos_idx] = 'interpolated'
                counters['case2'] += 2

            else:
                # Case 3: 합 > 0, 비인접 → NaN
                values[neg_idx] = np.nan
                values[pos_idx] = np.nan
                flags[neg_idx] = 'nan_case3'
                flags[pos_idx] = 'nan_case3'
                counters['case3'] += 2

        data_array[row_idx] = values

    df[HOUR_COLS] = data_array

    log_step(f'[2단계] 인접 양수 상쇄 처리 (PNNL 2.4.1): '
             f'Case1(NaN)={counters["case1"]:,}, '
             f'Case2(보간)={counters["case2"]:,}, '
             f'Case3(NaN)={counters["case3"]:,}')
    return df


def step3_no_offset_negative_to_nan(df):
    """
    3단계: 무상쇄 음수 NaN 처리 (PNNL Section 2.4.2)
    인접 양수가 없는 잔여 음수 → NaN 변환
    """
    n_converted = 0
    for col in HOUR_COLS:
        m = df[col] < 0
        n_converted += m.sum()
        if m.any():
            df.loc[m, col] = np.nan

    log_step(f'[3단계] 무상쇄 음수 NaN 처리 (PNNL 2.4.2): {n_converted:,}개 셀')
    return df


# ============================================================
# Phase 2: 결측 처리 (PNNL Section 3)
# ============================================================

def step4_daily_nan_filter(df):
    """
    4단계: 일자 단위 의심 데이터 비율 필터
    조건: 24시간 중 NaN ≥ 3개
    근거: PNNL Section 2.3 (원문 25% = 6개)
          데이터 분포 분석 결과 elbow at 3 (1→2→3에서 -99.5%→-81.2%→-76.4% 급감,
          3→4→5에서 -1.7%, +1.0% plateau)
    """
    nan_count = df[HOUR_COLS].isna().sum(axis=1)
    mask = nan_count >= DAILY_NAN_THRESHOLD
    n_removed = mask.sum()
    df = df[~mask].reset_index(drop=True)

    log_step(f'[4단계] 일자 NaN ≥ {DAILY_NAN_THRESHOLD}개 제거: '
             f'-{n_removed:,}행 (PNNL 2.3 + elbow 분석)')
    return df


def step5_consecutive_nan_filter(df):
    """
    5단계: 연속 NaN 길이 평가 (PNNL Section 3 Option 2)
      연속 NaN = 1 → 선형 보간 (PNNL Section 3.1)
      연속 NaN ≥ 2 → 일자 제거
    근거: PNNL Section 3 Option 2 원문 — "only ≤ 1 hour gaps be filled"
    """
    data_array = df[HOUR_COLS].values.astype(np.float32)

    rows_to_remove = []
    rows_to_interpolate = []

    for row_idx in range(len(data_array)):
        values = data_array[row_idx]
        is_nan = np.isnan(values)

        if not is_nan.any():
            continue

        # 최대 연속 NaN 길이 계산
        max_consec = 0
        current_consec = 0
        for v_is_nan in is_nan:
            if v_is_nan:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        if max_consec >= CONSEC_NAN_THRESHOLD:
            rows_to_remove.append(row_idx)
        elif max_consec == 1:
            rows_to_interpolate.append(row_idx)

    # 보간 적용 (연속 1개)
    n_interpolated = 0
    for row_idx in rows_to_interpolate:
        values = data_array[row_idx]
        interpolated = pd.Series(values).interpolate(
            method='linear', limit_direction='both'
        ).values
        # 보간 후 음수 clip
        interpolated = np.maximum(interpolated, 0)
        data_array[row_idx] = interpolated
        n_interpolated += 1

    df[HOUR_COLS] = data_array
    df_clean = df.drop(index=df.index[rows_to_remove]).reset_index(drop=True)

    log_step(f'[5단계] 연속 NaN 평가 (PNNL Section 3 Option 2): '
             f'제거(연속≥{CONSEC_NAN_THRESHOLD})={len(rows_to_remove):,}행, '
             f'보간(연속=1)={n_interpolated:,}행')
    return df_clean


# ============================================================
# Phase 3: 설비 단위 품질 필터
# ============================================================

def step6_min_observation_filter(df):
    """
    6단계: 관측 기간 < 1년 설비 제거
    근거: Quesada et al. (2024) — 계절 패턴 1사이클 필요
    """
    facility_span = df.groupby(FACILITY_COL)[DATE_COL].agg(['min', 'max'])
    facility_span['days'] = (facility_span['max'] - facility_span['min']).dt.days

    facilities_to_keep = facility_span[
        facility_span['days'] >= MIN_OBSERVATION_DAYS
    ].index

    n_before = df[FACILITY_COL].nunique()
    df = df[df[FACILITY_COL].isin(facilities_to_keep)].reset_index(drop=True)
    n_after = df[FACILITY_COL].nunique()

    log_step(f'[6단계] 관측 < {MIN_OBSERVATION_DAYS}일 설비 제거: '
             f'-{n_before - n_after:,}개 설비 (Quesada 2024)')
    return df


def step7_high_missing_facility_filter(df, facility_nan_rate):
    """
    7단계: 결측률 ≥ 28% 설비 제거
    근거: 데이터 분포 시각화 분석 — 27~28% 구간에 16개 → 28% 이후 0~1개 급감 (단절점)
    facility_nan_rate: 원본 데이터 기준 설비별 결측률 (pre-computed)
    """
    facilities_to_keep = facility_nan_rate[
        facility_nan_rate < MAX_MISSING_RATIO
    ].index

    n_before = df[FACILITY_COL].nunique()
    df = df[df[FACILITY_COL].isin(facilities_to_keep)].reset_index(drop=True)
    n_after = df[FACILITY_COL].nunique()

    log_step(f'[7단계] 결측률 ≥ {MAX_MISSING_RATIO * 100:.0f}% 설비 제거: '
             f'-{n_before - n_after:,}개 설비 (분포 단절점)')
    return df


def step8_high_processed_ratio_filter(df, facility_orig_size):
    """
    8단계: 처리 비율 ≥ 30% 설비 제거 (PNNL Section 2.3 응용)
    근거: PNNL 25% 가이드 + 데이터 분포에서 30~99% 구간이 거의 비어있음
    facility_orig_size: 원본 데이터 기준 설비별 행 수 (pre-computed)
    """
    orig_total_points = facility_orig_size * 24
    proc_valid_points = df.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].notna().sum().sum()
    )

    facility_stats = pd.DataFrame({'orig_total_points': orig_total_points})
    facility_stats['proc_valid_points'] = proc_valid_points.reindex(
        facility_stats.index
    ).fillna(0)
    facility_stats['processed_ratio'] = (
        1 - facility_stats['proc_valid_points'] / facility_stats['orig_total_points']
    ).clip(lower=0, upper=1)

    facilities_to_keep = facility_stats[
        facility_stats['processed_ratio'] < MAX_PROCESSED_RATIO
    ].index

    n_before = df[FACILITY_COL].nunique()
    df = df[df[FACILITY_COL].isin(facilities_to_keep)].reset_index(drop=True)
    n_after = df[FACILITY_COL].nunique()

    log_step(f'[8단계] 처리비율 ≥ {MAX_PROCESSED_RATIO * 100:.0f}% 설비 제거: '
             f'-{n_before - n_after:,}개 설비 (PNNL 2.3)')
    return df, facility_stats


# ============================================================
# Phase 4: 극단값 처리
# ============================================================

def step9_extreme_value_processing(df):
    """
    9단계: 종별 시간당 사용값 99.9%ile 기반 극단값 처리
      threshold × 100 이상 → 행 제거 (검침 오류)
      threshold 초과 ~ ×100 미만 → threshold로 clip
    근거: EDA에서 99.9%ile → 99.99%ile 사이 급격한 점프 확인
          99.9%ile의 100배 이상은 물리적으로 불가능한 수준 (검침기 오류)
    """
    # 종별 시간당 99.9%ile threshold 산출
    thresholds = {}
    for t in ALL_TYPES:
        mask = df['종별'] == t
        if mask.sum() == 0:
            continue
        all_vals = []
        for col in HOUR_COLS:
            v = df.loc[mask, col].dropna().values
            all_vals.append(v)
        all_vals = np.concatenate(all_vals)
        q999 = float(np.percentile(all_vals, EXTREME_PERCENTILE))
        thresholds[t] = round(q999, 4)
        print(f'    {t}: 99.9%ile = {q999:.4f} Gcal')
        del all_vals
    gc.collect()

    # 행 제거: threshold × 100 이상
    remove_mask = pd.Series(False, index=df.index)
    for t in thresholds:
        mask = df['종별'] == t
        th = thresholds[t]
        for col in HOUR_COLS:
            extreme = mask & (df[col] >= th * EXTREME_MULTIPLIER)
            remove_mask = remove_mask | extreme

    n_extreme = remove_mask.sum()
    if n_extreme > 0:
        types_removed = df.loc[remove_mask, '종별'].value_counts().to_dict()
        df = df[~remove_mask].reset_index(drop=True)
        print(f'    극단값 행 제거: {n_extreme:,}건 {types_removed}')

    # clip: threshold 초과값 → threshold
    total_clipped = 0
    for t in thresholds:
        mask = df['종별'] == t
        th = thresholds[t]
        for col in HOUR_COLS:
            m = mask & (df[col] > th)
            total_clipped += m.sum()
            if m.any():
                df.loc[m, col] = th

    log_step(f'[9단계] 극단값 처리: 행 제거 -{n_extreme:,}, '
             f'clip {total_clipped:,}셀 (99.9%ile × {EXTREME_MULTIPLIER})')
    return df, thresholds


# ============================================================
# 근거 분석 함수들 (--analyze 모드에서 실행)
# ============================================================

def analyze_step4_daily_nan_threshold(df, save_dir):
    """
    4단계 근거: 1~3단계 적용 후 일자별 NaN 개수 분포 분석
    → elbow at 3 확인 → DAILY_NAN_THRESHOLD = 3 결정 근거
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print('=' * 60)
    print('[4단계 근거] 일자별 NaN 개수 분포 분석')
    print('=' * 60)

    nan_count = df[HOUR_COLS].isna().sum(axis=1)

    # 기본 통계
    print(f'\n일자별 NaN 개수 기본 통계:')
    print(nan_count.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

    # NaN 개수별 빈도
    nan_dist = nan_count.value_counts().sort_index()
    nan_dist_df = pd.DataFrame({
        'nan_count': nan_dist.index,
        'days': nan_dist.values,
        'ratio_pct': (nan_dist.values / len(df)) * 100,
        'cumul_ratio_pct': (nan_dist.cumsum().values / len(df)) * 100
    })
    print(f'\nNaN 개수별 일자 분포:')
    print(nan_dist_df.to_string(index=False))

    # 감소율 분석 (elbow 탐지)
    print(f'\n감소율 분석 (elbow 탐지):')
    full_range = pd.Series(0, index=range(0, 25))
    full_range.update(nan_dist)
    print(f'{"NaN 개수":>8} | {"일수":>10} | {"변화 비율":>12}')
    print('-' * 40)
    for n in range(1, 13):
        days = full_range[n]
        prev_days = full_range[n - 1] if n > 0 else 0
        if prev_days > 0:
            change = (days - prev_days) / prev_days * 100
            print(f'{n:>8} | {days:>10,} | {change:>11.1f}%')
        else:
            print(f'{n:>8} | {days:>10,} | {"N/A":>12}')

    # 임계 후보별 제거 비율
    threshold_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    print(f'\n임계 후보별 제거 영향:')
    for t in threshold_candidates:
        n_removed = (nan_count >= t).sum()
        pct = n_removed / len(df) * 100
        marker = ' ← 채택' if t == DAILY_NAN_THRESHOLD else ''
        print(f'  NaN ≥ {t:>2}: {n_removed:>8,}행 ({pct:.4f}%){marker}')

    # 시각화 (4패널)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('4단계 근거: 일자별 NaN 개수 분포 분석', fontsize=14)

    # A: 전체 분포 (로그)
    ax = axes[0, 0]
    counts = nan_count.value_counts().sort_index()
    ax.bar(counts.index, counts.values, edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    for t, c, lw in [(3, 'green', 2), (6, 'red', 1), (12, 'orange', 1)]:
        ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=lw,
                   label=f'≥{t}개 ({t / 24 * 100:.0f}%)')
    ax.set_xlabel('일자별 NaN 개수')
    ax.set_ylabel('일수 (log)')
    ax.set_title('(a) 전체 분포 (로그 스케일)')
    ax.legend()
    ax.set_xticks(range(0, 25))

    # B: NaN≥1만 (자세히)
    ax = axes[0, 1]
    df_nz = nan_count[nan_count >= 1]
    if len(df_nz) > 0:
        counts_nz = df_nz.value_counts().sort_index()
        ax.bar(counts_nz.index, counts_nz.values, edgecolor='black', alpha=0.7)
        for t, c, lw in [(3, 'green', 2), (6, 'red', 1)]:
            ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=lw, label=f'≥{t}개')
        ax.set_xlabel('일자별 NaN 개수')
        ax.set_ylabel('일수')
        ax.set_title('(b) NaN ≥ 1 확대')
        ax.legend()
        ax.set_xticks(range(1, 25))

    # C: 누적 분포
    ax = axes[1, 0]
    ax.plot(nan_dist_df['nan_count'], nan_dist_df['cumul_ratio_pct'], 'o-', linewidth=2)
    for t, c in [(3, 'green'), (6, 'red')]:
        ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=1.5)
    ax.set_xlabel('일자별 NaN 개수')
    ax.set_ylabel('누적 비율 (%)')
    ax.set_title('(c) 누적 분포')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 25))

    # D: 임계별 제거 비율
    ax = axes[1, 1]
    removal_pcts = [(nan_count >= t).mean() * 100 for t in threshold_candidates]
    bars = ax.bar(threshold_candidates, removal_pcts, edgecolor='black', alpha=0.7)
    for bar, t in zip(bars, threshold_candidates):
        if t == DAILY_NAN_THRESHOLD:
            bar.set_color('green')
            bar.set_alpha(0.8)
    ax.set_xlabel('임계치 (NaN 개수)')
    ax.set_ylabel('제거 비율 (%)')
    ax.set_title(f'(d) 임계별 제거 영향 (녹색=채택 {DAILY_NAN_THRESHOLD})')
    ax.set_yscale('log')
    for t, r in zip(threshold_candidates, removal_pcts):
        ax.text(t, r, f'{r:.3f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_step4_nan_threshold.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n저장: {save_dir}/phase02_step4_nan_threshold.png')


def analyze_step7_nan_rate_breakpoint(df, save_dir):
    """
    7단계 근거: 설비별 결측률 분포 분석
    → 27~28% 구간 16개 → 28% 이후 0~1개 급감 (단절점)
    → MAX_MISSING_RATIO = 0.28 결정 근거
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print('\n' + '=' * 60)
    print('[7단계 근거] 설비별 결측률 분포 분석')
    print('=' * 60)

    nan_rate = df.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].isna().sum().sum() / (len(x) * 24)
    )
    total_inst = len(nan_rate)

    print(f'설비 수: {total_inst:,}')
    print(f'결측률 기본 통계:')
    print(nan_rate.describe())

    # ── Figure 1: 0~10% (1% 단위) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins_0_10 = np.arange(0, 11) / 100
    counts_0_10, _ = np.histogram(nan_rate, bins=bins_0_10)
    labels_0_10 = [str(i) for i in range(10)]
    axes[0].bar(labels_0_10, counts_0_10, color='steelblue', edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_0_10):
        axes[0].text(j, v + 30, str(v), ha='center', fontsize=9)
    axes[0].set_title('결측률 0~10% (1% 단위)')
    axes[0].set_xlabel('결측률 (%)')
    axes[0].set_ylabel('설비 수')

    bins_1_10 = np.arange(1, 11) / 100
    counts_1_10, _ = np.histogram(nan_rate, bins=bins_1_10)
    labels_1_10 = [f'{i}-{i + 1}' for i in range(1, 10)]
    axes[1].bar(labels_1_10, counts_1_10, color='steelblue', edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_1_10):
        axes[1].text(j, v + 5, str(v), ha='center', fontsize=9)
    axes[1].set_title('결측률 1~10% (1% 단위, 확대)')
    axes[1].set_xlabel('결측률 (%)')
    axes[1].set_ylabel('설비 수')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_nan_rate_1pct.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: elbow 확인 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins_0_30 = np.arange(0, 32, 2) / 100
    counts_0_30, _ = np.histogram(nan_rate, bins=bins_0_30)
    labels_0_30 = [f'{i}-{i + 2}' for i in range(0, 30, 2)]
    axes[0].bar(labels_0_30, counts_0_30, color='steelblue', edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_0_30):
        axes[0].text(j, v + 30, str(v), ha='center', fontsize=9)
    axes[0].set_title('결측률 0~30% 구간 (2% 단위)')
    axes[0].set_xlabel('결측률 (%)')
    axes[0].set_ylabel('설비 수')

    edges_pct = list(range(1, 31)) + [40, 50, 60, 80, 100]
    bins_1p = [e / 100 for e in edges_pct] + [1.01]
    counts_1p, _ = np.histogram(nan_rate, bins=bins_1p)
    labels_1p = [f'{edges_pct[i]}-{edges_pct[i + 1]}' if i < len(edges_pct) - 1
                 else f'{edges_pct[i]}-100' for i in range(len(edges_pct))]
    colors_1p = ['steelblue' if edges_pct[i] < 30 else 'darkorange'
                 for i in range(len(edges_pct))]
    axes[1].bar(range(len(counts_1p)), counts_1p, color=colors_1p,
                edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_1p):
        axes[1].text(j, v + 2, str(v), ha='center', fontsize=7)
    axes[1].set_xticks(range(len(labels_1p)))
    axes[1].set_xticklabels(labels_1p, rotation=45, ha='right', fontsize=7)
    axes[1].set_title('결측률 1%+ 구간 확대 (30%+ 빨간색)')
    axes[1].set_xlabel('결측률 (%)')
    axes[1].set_ylabel('설비 수')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_nan_rate_elbow.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: 단절점 확대 (핵심) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins_10_50 = np.arange(10, 52, 2) / 100
    counts_10_50, _ = np.histogram(nan_rate, bins=bins_10_50)
    labels_10_50 = [f'{i}-{i + 2}' for i in range(10, 50, 2)]
    axes[0].bar(labels_10_50, counts_10_50, color='steelblue', edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_10_50):
        axes[0].text(j, v + 0.3, str(v), ha='center', fontsize=9)
    axes[0].set_title('10~50% 구간 (2% 단위)')
    axes[0].set_xlabel('결측률 (%)')
    axes[0].set_ylabel('설비 수')
    axes[0].tick_params(axis='x', rotation=45)

    bins_20_40 = np.arange(20, 41) / 100
    counts_20_40, _ = np.histogram(nan_rate, bins=bins_20_40)
    labels_20_40 = [f'{i}-{i + 1}' for i in range(20, 40)]
    colors_20_40 = ['steelblue' if i < 28 else 'darkorange' for i in range(20, 40)]
    axes[1].bar(labels_20_40, counts_20_40, color=colors_20_40,
                edgecolor='black', linewidth=0.5)
    for j, v in enumerate(counts_20_40):
        axes[1].text(j, v + 0.2, str(v), ha='center', fontsize=9)
    axes[1].set_title('20~40% 구간 (1% 단위) — 단절점 확대')
    axes[1].set_xlabel('결측률 (%)')
    axes[1].set_ylabel('설비 수')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_nan_rate_breakpoint.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: 종합 분석 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(nan_rate, bins=50, color='steelblue', edgecolor='black', linewidth=0.3)
    axes[0].axvline(0.28, color='red', linestyle='--', label='채택 기준 (28%)')
    axes[0].axvline(0.50, color='orange', linestyle='--', label='선행연구 기준 (50%)')
    axes[0].set_title('설비별 결측률 분포 (전체)')
    axes[0].set_xlabel('결측률')
    axes[0].set_ylabel('설비 수')
    axes[0].legend(fontsize=8)

    over5 = nan_rate[nan_rate >= 0.05]
    axes[1].hist(over5, bins=30, color='darkorange', edgecolor='black', linewidth=0.3)
    axes[1].axvline(0.28, color='red', linestyle='--', label='28%')
    axes[1].axvline(0.50, color='orange', linestyle='--', label='50%')
    axes[1].set_title('결측률 5%+ 설비 확대 (elbow 확인)')
    axes[1].set_xlabel('결측률')
    axes[1].set_ylabel('설비 수')
    axes[1].legend(fontsize=8)

    thresholds_d = np.arange(0, 101)
    survival = [(nan_rate >= t / 100).sum() / total_inst * 100 for t in thresholds_d]
    axes[2].plot(thresholds_d, survival, color='steelblue', linewidth=1.5)
    pct_28 = (nan_rate >= 0.28).sum() / total_inst * 100
    pct_50 = (nan_rate >= 0.50).sum() / total_inst * 100
    axes[2].axvline(28, color='red', linestyle='--', label=f'28% ({pct_28:.2f}%)')
    axes[2].axvline(50, color='orange', linestyle='--', label=f'50% ({pct_50:.2f}%)')
    axes[2].set_title('결측률 역누적 분포')
    axes[2].set_xlabel('결측률 (%)')
    axes[2].set_ylabel('해당 결측률 이상 설비 비율 (%)')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_nan_rate_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'저장: {save_dir}/phase02_nan_rate_*.png (4개)')


def analyze_step8_processed_ratio(df_processed, df_original, save_dir):
    """
    8단계 근거: 설비별 처리 비율 분포 분석
    → 30~99% 구간이 거의 비어있음 → MAX_PROCESSED_RATIO = 0.30 결정 근거
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print('\n' + '=' * 60)
    print('[8단계 근거] 설비별 처리 비율 분포 분석')
    print('=' * 60)

    # 처리 비율 계산
    orig_total = df_original.groupby(FACILITY_COL).size() * 24
    proc_valid = df_processed.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].notna().sum().sum()
    )

    stats = pd.DataFrame({'orig': orig_total})
    stats['valid'] = proc_valid.reindex(stats.index).fillna(0)
    stats['ratio'] = (1 - stats['valid'] / stats['orig']).clip(0, 1)

    print(f'\n처리 비율 기본 통계:')
    print(stats['ratio'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

    # 임계 후보별 영향
    print(f'\n임계 후보별 설비 제거 영향:')
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n = (stats['ratio'] >= t).sum()
        pct = n / len(stats) * 100
        marker = ' ← 채택' if abs(t - MAX_PROCESSED_RATIO) < 0.01 else ''
        print(f'  ≥{t * 100:>4.0f}%: {n:>5}개 ({pct:.2f}%){marker}')

    # 시각화 (4패널)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('8단계 근거: 설비별 처리 비율 분포 분석', fontsize=14)

    # A: 전체 분포
    ax = axes[0, 0]
    ax.hist(stats['ratio'], bins=100, edgecolor='black', alpha=0.7)
    for t, c, lw in [(0.25, 'red', 1), (0.30, 'green', 2), (0.50, 'purple', 1)]:
        ax.axvline(t, color=c, linestyle='--', linewidth=lw, label=f'{int(t * 100)}%')
    ax.set_xlabel('처리 비율')
    ax.set_ylabel('설비 수')
    ax.set_title('(a) 전체 분포')
    ax.legend()

    # B: 로그 스케일
    ax = axes[0, 1]
    ax.hist(stats['ratio'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    for t, c, lw in [(0.25, 'red', 1), (0.30, 'green', 2), (0.50, 'purple', 1)]:
        ax.axvline(t, color=c, linestyle='--', linewidth=lw, label=f'{int(t * 100)}%')
    ax.set_xlabel('처리 비율')
    ax.set_ylabel('설비 수 (log)')
    ax.set_title('(b) 로그 스케일')
    ax.legend()

    # C: CDF
    ax = axes[1, 0]
    sorted_r = np.sort(stats['ratio'].dropna().values)
    cumul = np.arange(1, len(sorted_r) + 1) / len(sorted_r) * 100
    ax.plot(sorted_r, cumul, linewidth=2)
    for t, c in [(0.25, 'red'), (0.30, 'green')]:
        ax.axvline(t, color=c, linestyle='--')
    ax.set_xlabel('처리 비율')
    ax.set_ylabel('누적 비율 (%)')
    ax.set_title('(c) CDF')
    ax.grid(True, alpha=0.3)

    # D: 임계별 영향
    ax = axes[1, 1]
    thresholds_eval = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    remove_pcts = [(stats['ratio'] >= t).mean() * 100 for t in thresholds_eval]
    bars = ax.bar([int(t * 100) for t in thresholds_eval], remove_pcts,
                  edgecolor='black', alpha=0.7, width=3)
    for bar, t in zip(bars, thresholds_eval):
        if abs(t - MAX_PROCESSED_RATIO) < 0.01:
            bar.set_color('green')
    ax.set_xlabel('임계치 (%)')
    ax.set_ylabel('제거 비율 (%)')
    ax.set_title(f'(d) 임계별 영향 (녹색=채택 {int(MAX_PROCESSED_RATIO * 100)}%)')
    for t, r in zip(thresholds_eval, remove_pcts):
        ax.text(int(t * 100), r, f'{r:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase02_step8_processed_ratio.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'저장: {save_dir}/phase02_step8_processed_ratio.png')


# ============================================================
# 메인 파이프라인
# ============================================================

def run_preprocess():
    """전처리 9단계 실행"""
    t_start = time.time()

    print('=' * 60)
    print('Phase 2. 정밀 전처리 (PNNL + Quesada 기반 9단계)')
    print('=' * 60)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    dst = f'{ARCHIVE_DIR}/all_data_phase0.parquet'
    if not os.path.exists(dst):
        shutil.copy2(f'{PROCESSED_DIR}/all_data.parquet', dst)
        print(f'  Archived -> {dst}')

    import pyarrow.parquet as pq
    _table = pq.read_table(f'{PROCESSED_DIR}/all_data.parquet')
    df = _table.to_pandas(self_destruct=True)
    del _table
    gc.collect()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    initial_rows = len(df)
    initial_facilities = df[FACILITY_COL].nunique()
    print(f'  원본: {initial_rows:,}행, {initial_facilities:,}개 설비')

    # 메모리 최적화
    for col in HOUR_COLS:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    if '총사용량' in df.columns:
        df['총사용량'] = df['총사용량'].astype('float32')
    for col in ['종별', '지사', '계절']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    gc.collect()

    # 냉방시즌 컬럼 (냉수용 전용)
    df['냉방시즌'] = df['월'].isin([6, 7, 8, 9])

    # 원본 통계 사전 계산 (7/8단계 및 품질 검증에서 필요 — 전체 복사 대신 집계값만 보관)
    print('  설비별 원본 통계 사전 계산...')
    facility_nan_rate = df.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].isna().sum().sum() / (len(x) * 24)
    )
    facility_orig_size = df.groupby(FACILITY_COL).size()
    _original_type_counts = {t: int((df['종별'] == t).sum()) for t in ALL_TYPES}
    gc.collect()

    # ═══ Phase 1: 음수 처리 ═══
    print('\n' + '=' * 60)
    print('[Phase 1] 음수 처리 (PNNL Section 2.4)')
    print('=' * 60)

    df = step1_remove_daily_negative_total(df)
    df = step2_offset_negative_processing(df)
    df = step3_no_offset_negative_to_nan(df)
    gc.collect()

    # ═══ Phase 2: 결측 처리 ═══
    print('\n' + '=' * 60)
    print('[Phase 2] 결측 처리 (PNNL Section 3)')
    print('=' * 60)

    df = step4_daily_nan_filter(df)
    df = step5_consecutive_nan_filter(df)
    gc.collect()

    # ═══ Phase 3: 설비 필터 ═══
    print('\n' + '=' * 60)
    print('[Phase 3] 설비 단위 품질 필터')
    print('=' * 60)

    df = step6_min_observation_filter(df)
    df = step7_high_missing_facility_filter(df, facility_nan_rate)
    df, facility_stats = step8_high_processed_ratio_filter(df, facility_orig_size)
    gc.collect()

    # ═══ Phase 4: 극단값 ═══
    print('\n' + '=' * 60)
    print('[Phase 4] 극단값 처리')
    print('=' * 60)

    df, thresholds = step9_extreme_value_processing(df)
    gc.collect()

    # ═══ 후처리 ═══
    print('\n' + '=' * 60)
    print('[후처리]')
    print('=' * 60)

    # 총사용량 재계산
    df['총사용량'] = df[HOUR_COLS].sum(axis=1)
    log_step('총사용량 재계산')

    # 정규화 파라미터 산출
    print('\n  정규화 파라미터 산출 (종별×시즌 8그룹)...')
    norm_params = {}
    cols = HOUR_COLS + ['총사용량']

    for t in HEATING_TYPES:
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
            print(f'    {key}: N={mask.sum():>9,}')

    for season in [True, False]:
        mask = (df['종별'] == '냉수용') & (df['냉방시즌'] == season)
        s_label = '냉방' if season else '비냉방'
        key = f'냉수용_{s_label}'
        means = df.loc[mask, cols].mean()
        stds = df.loc[mask, cols].std().replace(0, 1)
        norm_params[key] = {
            'mean': {k: round(float(v), 6) for k, v in means.items()},
            'std': {k: round(float(v), 6) for k, v in stds.items()},
            'n_rows': int(mask.sum())
        }
        print(f'    {key}: N={mask.sum():>9,}')

    with open(f'{PROCESSED_DIR}/norm_params.json', 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, ensure_ascii=False, indent=2)
    log_step('정규화 파라미터 저장 (norm_params.json)')

    # ═══ 품질 검증 ═══
    print('\n' + '=' * 60)
    print('[품질 검증]')
    print('=' * 60)

    checks = {}
    checks['neg_hourly'] = int((df[HOUR_COLS] < 0).sum().sum())
    checks['nan_hourly'] = int(df[HOUR_COLS].isnull().sum().sum())
    checks['neg_total'] = int((df['총사용량'] < 0).sum())

    # 원본 종별 행 수 (facility_orig_size에서 복원 불가 — 별도 계산 필요)
    # initial_rows에서 종별 비율로 역산 대신, 사전 보관된 변수 사용
    original_counts = _original_type_counts

    checks['removal_rates'] = {}
    for t in ALL_TYPES:
        current = int((df['종별'] == t).sum())
        orig = original_counts[t]
        rate = round((1 - current / orig) * 100, 2) if orig > 0 else 0
        checks['removal_rates'][t] = {
            'original': orig, 'current': current, 'rate': rate
        }

    for k, v in checks.items():
        if k == 'removal_rates':
            for t, d in v.items():
                status = 'PASS' if d['rate'] < 5 else 'WARN'
                print(f'  [{status}] {t} 제거율: {d["rate"]}% ({d["original"]:,} → {d["current"]:,})')
        else:
            status = 'PASS' if v == 0 else 'FAIL'
            print(f'  [{status}] {k}: {v}')

    # ═══ 저장 ═══
    print('\n저장 중...')
    output_path = f'{PROCESSED_DIR}/all_data_clean.parquet'
    df.to_parquet(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 ** 2)

    summary = {
        'initial_rows': initial_rows,
        'initial_facilities': initial_facilities,
        'final_rows': len(df),
        'final_facilities': df[FACILITY_COL].nunique(),
        'preservation_rate': round(len(df) / initial_rows * 100, 1),
        'thresholds': thresholds,
        'log': log,
        'checks': checks,
    }
    with open(f'{PROCESSED_DIR}/phase2_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    facility_stats.to_csv(
        f'{PROCESSED_DIR}/facility_processed_ratio.csv', encoding='utf-8-sig'
    )

    # ═══ 최종 요약 ═══
    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'전처리 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    for entry in log:
        print(f'  {entry}')
    print(f'\n  행: {initial_rows:,} → {len(df):,} ({summary["preservation_rate"]}% 유지)')
    print(f'  설비: {initial_facilities:,} → {df[FACILITY_COL].nunique():,}')
    print(f'  파일: {output_path} ({file_size:.1f} MB)')
    print(f'  파라미터: {PROCESSED_DIR}/norm_params.json')
    print(f'  요약: {PROCESSED_DIR}/phase2_summary.json')


def run_analyze():
    """근거 분석만 실행 (시각화 생성)"""
    os.makedirs(FIG_DIR, exist_ok=True)

    print('=' * 60)
    print('근거 분석 모드 — 임계치 결정 시각화 생성')
    print('=' * 60)

    # 원본 데이터 로드
    print('\n원본 데이터 로딩...')
    df = pd.read_parquet(f'{PROCESSED_DIR}/all_data.parquet')
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df_original = df.copy()
    print(f'  {len(df):,}행, {df[FACILITY_COL].nunique():,}개 설비')

    # 7단계 근거: NaN 28% 단절점
    analyze_step7_nan_rate_breakpoint(df, FIG_DIR)

    # 1~3단계 적용 (4단계 분석용)
    print('\n\nPhase 1 적용 중 (4단계 분석을 위해)...')
    df = step1_remove_daily_negative_total(df)
    df = step2_offset_negative_processing(df)
    df = step3_no_offset_negative_to_nan(df)

    # 4단계 근거: NaN 개수 elbow
    analyze_step4_daily_nan_threshold(df, FIG_DIR)

    # 4~7단계 적용 (8단계 분석용)
    print('\n\nPhase 2~3 적용 중 (8단계 분석을 위해)...')
    df = step4_daily_nan_filter(df)
    df = step5_consecutive_nan_filter(df)
    df = step6_min_observation_filter(df)
    df = step7_high_missing_facility_filter(df, df_original)

    # 8단계 근거: 처리비율 분포
    analyze_step8_processed_ratio(df, df_original, FIG_DIR)

    print('\n' + '=' * 60)
    print('근거 분석 완료')
    print(f'시각화 저장 위치: {FIG_DIR}/')
    print('=' * 60)

    del df_original
    gc.collect()


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2 전처리 파이프라인')
    parser.add_argument('--analyze', action='store_true',
                        help='근거 분석만 실행 (시각화 생성)')
    parser.add_argument('--all', action='store_true',
                        help='근거 분석 + 전처리 모두 실행')
    args = parser.parse_args()

    if args.all:
        run_analyze()
        print('\n\n')
        log.clear()
        run_preprocess()
    elif args.analyze:
        run_analyze()
    else:
        run_preprocess()
