import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0. 설정 및 데이터 로드
# ============================================================
print("데이터 불러오는 중")
file_path = r"C:\Users\kjw31\OneDrive\Desktop\Heat-Anomaly-Detection\data\processed\all_data.parquet"
df = pd.read_parquet(file_path)

print("데이터 크기:", df.shape)
print(df.info())

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SAVE_DIR = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\neg"
os.makedirs(SAVE_DIR, exist_ok=True)

HOUR_COLS = [f"{i}시" for i in range(1, 25)]
FACILITY_COL = "설치"  
DATE_COL = "날짜"


# ============================================================
# 1~3단계 적용 (간단 버전 — 4단계 임계 검증용)
# ============================================================
def apply_phase1_steps(df):
    """Phase 1 (1~3단계) 적용 — 최적화 버전"""
    df = df.copy()
    
    # ── 1단계: 일별 총사용량 음수 행 제거 ──
    daily_sum = df[HOUR_COLS].sum(axis=1, skipna=True)
    n_step1 = (daily_sum < 0).sum()
    df = df[daily_sum >= 0].reset_index(drop=True)  # ★ 인덱스 리셋
    print(f"[1단계] 일별 총사용량 음수 행 제거: {n_step1:,}행")
    print(f"        잔여: {len(df):,}행\n")
    
    # ── 2단계: 인접 양수 상쇄 음수 처리 (PNNL Section 2.4.1) ──
    OFFSET_WINDOW = 2
    counters = {'case1': 0, 'case2': 0, 'case3': 0}
    
    # ★ 핵심 최적화 1: 전체 데이터를 numpy 배열로 한 번만 추출
    data_array = df[HOUR_COLS].values.astype(float)  # (n_rows, 24)
    
    # ★ 핵심 최적화 2: 음수가 있는 행만 식별
    has_negative = (data_array < 0).any(axis=1)
    rows_with_neg = np.where(has_negative)[0]
    
    print(f"[2단계] 음수 포함 행: {len(rows_with_neg):,}개 (전체 {len(df):,}개 중)")
    print(f"        → 이 행들만 순회하여 처리")
    
    # ★ 핵심 최적화 3: 해당 행만 순회
    for row_idx in rows_with_neg:
        values = data_array[row_idx].copy()
        flags = ['collected'] * len(values)
        
        neg_positions = np.where(
            ~np.isnan(values) & (values < 0)
        )[0]
        
        for neg_idx in neg_positions:
            if flags[neg_idx] != 'collected':
                continue
            
            neg_value = values[neg_idx]
            
            # 인접 윈도우 내 양수 탐색
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
                # Case 1
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
                # Case 2
                avg_value = sum_value / (duration + 1)
                values[neg_idx] = avg_value
                values[pos_idx] = avg_value
                flags[neg_idx] = 'interpolated'
                flags[pos_idx] = 'interpolated'
                counters['case2'] += 2
            
            else:
                # Case 3
                values[neg_idx] = np.nan
                values[pos_idx] = np.nan
                flags[neg_idx] = 'nan_case3'
                flags[pos_idx] = 'nan_case3'
                counters['case3'] += 2
        
        # ★ numpy 배열에 직접 쓰기 (df.loc 안 씀)
        data_array[row_idx] = values
    
    # ★ 한 번에 DataFrame에 다시 할당
    df[HOUR_COLS] = data_array
    
    print(f"        Case 1 (NaN): {counters['case1']}, "
          f"Case 2 (보간): {counters['case2']}, "
          f"Case 3 (NaN): {counters['case3']}\n")
    
    # ── 3단계: 무상쇄 음수 NaN 처리 (PNNL Section 2.4.2) ──
    n_step3 = (df[HOUR_COLS] < 0).sum().sum()
    df[HOUR_COLS] = df[HOUR_COLS].where(~(df[HOUR_COLS] < 0), np.nan)
    print(f"[3단계] 무상쇄 음수 NaN 처리: {n_step3:,}개\n")
    
    return df


# ============================================================
# 4단계 임계 검증 — 일자별 NaN 개수 분포 분석
# ============================================================

def analyze_daily_nan_distribution(df, save_dir):
    """
    1~3단계 적용 후 일자별 NaN 개수 분포 분석
    """
    print("=" * 60)
    print("[4단계 임계 검증] 일자별 NaN 개수 분포 분석")
    print("=" * 60)
    
    # 일자별 NaN 개수 계산
    df['nan_count'] = df[HOUR_COLS].isna().sum(axis=1)
    
    # ────────────────────────────────────
    # (1) 기본 통계
    # ────────────────────────────────────
    print("\n[1] 일자별 NaN 개수 기본 통계")
    stats = df['nan_count'].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    )
    print(stats)
    
    stats.to_csv(
        os.path.join(save_dir, "01_daily_nan_count_stats.csv"),
        encoding="utf-8-sig"
    )
    
    # ────────────────────────────────────
    # (2) NaN 개수별 빈도
    # ────────────────────────────────────
    print("\n[2] NaN 개수별 일자 분포")
    nan_dist = df['nan_count'].value_counts().sort_index()
    nan_dist_df = pd.DataFrame({
        'nan_count': nan_dist.index,
        'days': nan_dist.values,
        'ratio_pct': (nan_dist.values / len(df)) * 100,
        'cumul_ratio_pct': (nan_dist.cumsum().values / len(df)) * 100
    })
    print(nan_dist_df)
    
    nan_dist_df.to_csv(
        os.path.join(save_dir, "02_daily_nan_count_distribution.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    
    # ────────────────────────────────────
    # (3) 임계 후보별 제거 비율
    # ────────────────────────────────────
    print("\n[3] 임계 후보별 일자 제거 영향")
    threshold_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    threshold_eval = pd.DataFrame({
        'threshold': threshold_candidates,
        'pct_of_24h': [t/24*100 for t in threshold_candidates],
        'days_removed': [(df['nan_count'] >= t).sum() 
                        for t in threshold_candidates],
        'ratio_pct': [(df['nan_count'] >= t).mean() * 100 
                     for t in threshold_candidates]
    })
    print(threshold_eval)
    
    threshold_eval.to_csv(
        os.path.join(save_dir, "03_threshold_evaluation.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    
    # ────────────────────────────────────
    # (4) 시각화 1: 전체 분포 (로그 스케일)
    # ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 4-A: 전체 NaN 개수 분포 (로그)
    ax = axes[0, 0]
    counts = df['nan_count'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    
    # 임계 후보 표시
    for t, c, lw in [(3, 'green', 1), (6, 'red', 2), 
                      (12, 'orange', 1)]:
        ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=lw, 
                   label=f'≥{t}개 ({t/24*100:.0f}%)')
    
    ax.set_xlabel('일자별 NaN 개수')
    ax.set_ylabel('일수 (log scale)')
    ax.set_title('일자별 NaN 개수 분포 (전체, 로그 스케일)')
    ax.legend()
    ax.set_xticks(range(0, 25))
    
    # 4-B: NaN 개수 1 이상만 (자세히)
    ax = axes[0, 1]
    df_nonzero = df[df['nan_count'] >= 1]
    if len(df_nonzero) > 0:
        counts_nz = df_nonzero['nan_count'].value_counts().sort_index()
        ax.bar(counts_nz.index, counts_nz.values, 
               edgecolor='black', alpha=0.7)
        
        for t, c, lw in [(3, 'green', 1), (6, 'red', 2), 
                          (12, 'orange', 1)]:
            ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=lw,
                       label=f'≥{t}개')
        
        ax.set_xlabel('일자별 NaN 개수')
        ax.set_ylabel('일수')
        ax.set_title('일자별 NaN 개수 분포 (NaN ≥ 1만)')
        ax.legend()
        ax.set_xticks(range(1, 25))
    
    # 4-C: 누적 비율 곡선
    ax = axes[1, 0]
    cumul = nan_dist_df['cumul_ratio_pct'].values
    ax.plot(nan_dist_df['nan_count'], cumul, 'o-', linewidth=2)
    
    for t, c in [(3, 'green'), (6, 'red'), (12, 'orange')]:
        ax.axvline(t - 0.5, color=c, linestyle='--', linewidth=1.5)
        if t in nan_dist_df['nan_count'].values:
            y_val = nan_dist_df[nan_dist_df['nan_count'] == t][
                'cumul_ratio_pct'].values[0]
            ax.annotate(f'{t}개: {y_val:.4f}%',
                       xy=(t, y_val), xytext=(t+1, y_val-5),
                       fontsize=9)
    
    ax.set_xlabel('일자별 NaN 개수')
    ax.set_ylabel('누적 비율 (%)')
    ax.set_title('NaN 개수별 누적 분포')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 25))
    
    # 4-D: 임계 후보별 제거 일자 비율
    ax = axes[1, 1]
    bars = ax.bar(threshold_eval['threshold'], 
                   threshold_eval['ratio_pct'],
                   edgecolor='black', alpha=0.7)
    
    # 6 (PNNL 25%) 강조
    for bar, t in zip(bars, threshold_eval['threshold']):
        if t == 6:
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    ax.set_xlabel('임계치 (NaN 개수)')
    ax.set_ylabel('제거되는 일자 비율 (%)')
    ax.set_title('임계 후보별 제거 영향 (빨강=PNNL 25%)')
    ax.set_yscale('log')
    
    # 값 표시
    for i, (t, r) in enumerate(zip(threshold_eval['threshold'], 
                                    threshold_eval['ratio_pct'])):
        ax.text(t, r, f'{r:.4f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_distribution_analysis.png"), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # ────────────────────────────────────
    # (5) Gap/Elbow 분석
    # ────────────────────────────────────
    print("\n[4] 분포의 Gap/Elbow 분석")
    
    # 인접 NaN 개수 간 빈도 차이 (gap 탐지)
    nan_dist_full = nan_dist_df.set_index('nan_count')['days']
    
    # 0 ~ 24까지 채우기 (없는 값은 0)
    full_range = pd.Series(0, index=range(0, 25))
    full_range.update(nan_dist_full)
    
    # 빈도 비율 변화 (감소율)
    print("\nNaN 개수 1 → N으로 갈 때의 일수 변화:")
    print(f"{'NaN 개수':>8} | {'일수':>10} | {'변화 비율':>12}")
    print("-" * 40)
    for n in range(1, 13):
        days = full_range[n]
        prev_days = full_range[n-1] if n > 0 else 0
        if prev_days > 0:
            change = (days - prev_days) / prev_days * 100
            print(f"{n:>8} | {days:>10,} | {change:>11.1f}%")
        else:
            print(f"{n:>8} | {days:>10,} | {'N/A':>12}")
    
# ============================================================
# 실행
# ============================================================

def main():
    # 데이터 로드
    print("데이터 로딩 중...")
    file_path = r"C:\Users\kjw31\OneDrive\Desktop\Heat-Anomaly-Detection\data\processed\all_data.parquet"
    df = pd.read_parquet(file_path)
    print(f"원본 데이터: {df.shape}\n")
    
    # 1~3단계 적용
    print("=" * 60)
    print("[Phase 1] 음수 처리 (1~3단계 적용)")
    print("=" * 60 + "\n")
    
    df_processed = apply_phase1_steps(df)
    
    # 중간 결과 저장
    df_processed.to_parquet(
        os.path.join(SAVE_DIR, "after_phase1.parquet")
    )
    print(f"Phase 1 적용 후 데이터: {df_processed.shape}")
    print(f"저장: {os.path.join(SAVE_DIR, 'after_phase1.parquet')}\n")
    
    # 4단계 임계 검증
    df_with_count, nan_dist_df, threshold_eval = \
        analyze_daily_nan_distribution(df_processed, SAVE_DIR)
    
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
    print(f"저장 경로: {SAVE_DIR}")
    print("""
저장된 파일:
- 01_daily_nan_count_stats.csv: 기본 통계
- 02_daily_nan_count_distribution.csv: NaN 개수별 분포
- 03_threshold_evaluation.csv: 임계 후보별 영향
- 04_distribution_analysis.png: 시각화
- after_phase1.parquet: Phase 1 적용 후 데이터
    """)

"""
Phase 2 (4~5단계) + Phase 3 (6~7단계) 적용 후
8단계 임계 결정을 위한 처리 비율 분포 분석

전제: Phase 1 (1~3단계) 적용 완료된 데이터 사용
"""


SAVE_DIR = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\nan"
os.makedirs(SAVE_DIR, exist_ok=True)

# 임계치
DAILY_NAN_THRESHOLD = 3        # 4단계: 분포 분석 결과 기반
CONSEC_NAN_THRESHOLD = 2       # 5단계: PNNL Option 2 (1시간 초과)
MIN_OBSERVATION_DAYS = 365     # 6단계: Quesada et al. 2024
MAX_MISSING_RATIO = 0.28       # 7단계: 데이터 분포 분석


# ============================================================
# Phase 2: 결측 처리
# ============================================================

def step4_daily_nan_filter(df):
    """
    4단계: 일자 단위 의심 데이터 비율 필터
    조건: 24시간 중 NaN ≥ 3개 (분포 분석 기반)
    """
    df = df.copy()
    nan_count = df[HOUR_COLS].isna().sum(axis=1)
    n_removed = (nan_count >= DAILY_NAN_THRESHOLD).sum()
    
    df_clean = df[nan_count < DAILY_NAN_THRESHOLD].reset_index(drop=True)
    
    print(f"[4단계] 일자 단위 NaN ≥ {DAILY_NAN_THRESHOLD}개 제거")
    print(f"        제거 일자: {n_removed:,}일")
    print(f"        잔여: {len(df_clean):,}일\n")
    
    return df_clean


def step5_consecutive_nan_filter(df):
    """
    5단계: 연속 NaN 길이 평가 (PNNL Section 3 Option 2)
    - 연속 NaN = 1 → 선형 보간
    - 연속 NaN ≥ 2 → 일자 제거
    """
    df = df.copy()
    data_array = df[HOUR_COLS].values.astype(float)
    
    # 각 행의 최대 연속 NaN 길이 계산 (벡터화)
    max_consec_nan = np.zeros(len(data_array), dtype=int)
    n_interpolated = 0
    
    rows_to_remove = []
    rows_to_interpolate = []
    
    for row_idx in range(len(data_array)):
        values = data_array[row_idx]
        is_nan = np.isnan(values)
        
        if not is_nan.any():
            continue  # NaN 없음, 스킵
        
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
    
    # 보간 적용
    for row_idx in rows_to_interpolate:
        values = data_array[row_idx]
        interpolated = pd.Series(values).interpolate(
            method='linear', limit_direction='both'
        ).values
        data_array[row_idx] = interpolated
        n_interpolated += 1
    
    # DataFrame에 반영
    df[HOUR_COLS] = data_array
    
    # 연속 NaN ≥ 2 일자 제거
    df_clean = df.drop(index=rows_to_remove).reset_index(drop=True)
    
    print(f"[5단계] 연속 NaN 길이 평가 (PNNL Section 3 Option 2)")
    print(f"        제거 일자 (연속 ≥ {CONSEC_NAN_THRESHOLD}): {len(rows_to_remove):,}일")
    print(f"        보간 일자 (연속 = 1): {n_interpolated:,}일")
    print(f"        잔여: {len(df_clean):,}일\n")
    
    return df_clean


# ============================================================
# Phase 3: 설비 단위 품질 필터
# ============================================================

def step6_min_observation_filter(df):
    """
    6단계: 관측 기간 < 1년 설비 제거
    근거: Quesada et al. (2024)
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # 설비별 관측 기간 계산
    facility_span = df.groupby(FACILITY_COL)[DATE_COL].agg(['min', 'max'])
    facility_span['days'] = (facility_span['max'] - facility_span['min']).dt.days
    
    facilities_to_keep = facility_span[
        facility_span['days'] >= MIN_OBSERVATION_DAYS
    ].index
    
    n_before = df[FACILITY_COL].nunique()
    df_clean = df[df[FACILITY_COL].isin(facilities_to_keep)].copy()
    n_after = df_clean[FACILITY_COL].nunique()
    
    print(f"[6단계] 관측 기간 < {MIN_OBSERVATION_DAYS}일 설비 제거 (Quesada 2024)")
    print(f"        제거 설비: {n_before - n_after:,}개")
    print(f"        잔여: {n_after:,}개 설비, {len(df_clean):,}일\n")
    
    return df_clean


def step7_high_missing_facility_filter(df_processed, df_original):
    """
    7단계: 결측률 ≥ 28% 설비 제거
    
    Note: 결측률은 원본 데이터(전처리 전) 기준
    """
    # 원본 기준 설비별 결측률
    facility_missing = df_original.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].isna().sum().sum() / (len(x) * 24)
    )
    
    facilities_to_keep = facility_missing[
        facility_missing < MAX_MISSING_RATIO
    ].index
    
    n_before = df_processed[FACILITY_COL].nunique()
    df_clean = df_processed[
        df_processed[FACILITY_COL].isin(facilities_to_keep)
    ].copy().reset_index(drop=True)
    n_after = df_clean[FACILITY_COL].nunique()
    
    print(f"[7단계] 결측률 ≥ {MAX_MISSING_RATIO*100:.0f}% 설비 제거")
    print(f"        제거 설비: {n_before - n_after:,}개")
    print(f"        잔여: {n_after:,}개 설비, {len(df_clean):,}일\n")
    
    return df_clean


# ============================================================
# 8단계 임계 검증 — 설비별 처리 비율 분포 분석
# ============================================================

def analyze_facility_processed_ratio(df_processed, df_original, save_dir):
    """
    설비별 처리/추정 비율 분포 분석
    
    처리 비율 = (1~7단계에서 처리/제거된 시점 수) / (원본 전체 시점 수)
    
    이는 PNNL Section 2.3 원문의 "processed or estimated" 비율에 해당.
    """
    print("=" * 60)
    print("[8단계 임계 검증] 설비별 처리 비율 분포 분석")
    print("=" * 60)
    
    # ─── 원본 기준 설비별 전체 시점 수 ───
    orig_grouped = df_original.groupby(FACILITY_COL)
    orig_total_points = orig_grouped.size() * 24  # 일수 × 24시간
    
    # ─── 처리 후 설비별 유효 시점 수 (NaN이 아닌 값) ───
    proc_grouped = df_processed.groupby(FACILITY_COL)
    
    facility_stats = pd.DataFrame({
        'orig_total_points': orig_total_points
    })
    
    # 처리 후 살아남은 유효 시점 (NaN 제외)
    proc_valid_points = proc_grouped.apply(
        lambda x: x[HOUR_COLS].notna().sum().sum()
    )
    
    # 처리 후 데이터에 없는 설비는 0으로 (모두 제거된 설비)
    facility_stats['proc_valid_points'] = proc_valid_points
    facility_stats['proc_valid_points'] = facility_stats['proc_valid_points'].fillna(0)
    
    # 처리/추정된 시점 = 원본 - 살아남은 유효
    facility_stats['processed_points'] = (
        facility_stats['orig_total_points'] - facility_stats['proc_valid_points']
    )
    
    # 처리 비율
    facility_stats['processed_ratio'] = (
        facility_stats['processed_points'] / facility_stats['orig_total_points']
    )
    
    # 처리 비율이 1을 넘는 경우(보간 등으로 인한 오류) 클리핑
    facility_stats['processed_ratio'] = facility_stats['processed_ratio'].clip(
        lower=0, upper=1
    )
    
    # ─────────────────────────────────────
    # (1) 기본 통계
    # ─────────────────────────────────────
    print("\n[1] 설비별 처리 비율 기본 통계")
    stats = facility_stats['processed_ratio'].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    )
    print(stats)
    
    stats.to_csv(
        os.path.join(save_dir, "01_facility_processed_ratio_stats.csv"),
        encoding="utf-8-sig"
    )
    
    facility_stats.to_csv(
        os.path.join(save_dir, "02_facility_processed_ratio.csv"),
        encoding="utf-8-sig"
    )
    
    # ─────────────────────────────────────
    # (2) 임계 후보별 제거 영향
    # ─────────────────────────────────────
    print("\n[2] 임계 후보별 설비 제거 영향")
    threshold_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    threshold_eval = pd.DataFrame({
        'threshold': threshold_candidates,
        'threshold_pct': [t*100 for t in threshold_candidates],
        'facilities_removed': [
            (facility_stats['processed_ratio'] >= t).sum() 
            for t in threshold_candidates
        ],
        'remove_ratio_pct': [
            (facility_stats['processed_ratio'] >= t).mean() * 100 
            for t in threshold_candidates
        ]
    })
    print(threshold_eval)
    
    threshold_eval.to_csv(
        os.path.join(save_dir, "03_threshold_evaluation.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    
    # ─────────────────────────────────────
    # (3) 시각화
    # ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 3-A: 전체 분포 (선형 스케일)
    ax = axes[0, 0]
    ax.hist(facility_stats['processed_ratio'], bins=100, 
            edgecolor='black', alpha=0.7)
    
    for t, c, lw in [(0.10, 'green', 1), (0.20, 'blue', 1),
                      (0.25, 'red', 2), (0.30, 'orange', 1),
                      (0.50, 'purple', 1)]:
        ax.axvline(t, color=c, linestyle='--', linewidth=lw,
                   label=f'{int(t*100)}%')
    
    ax.set_xlabel('설비별 처리 비율')
    ax.set_ylabel('설비 수')
    ax.set_title('설비별 처리 비율 분포 (전체)')
    ax.legend()
    
    # 3-B: 로그 스케일 (꼬리 자세히)
    ax = axes[0, 1]
    ax.hist(facility_stats['processed_ratio'], bins=100, 
            edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    
    for t, c, lw in [(0.10, 'green', 1), (0.20, 'blue', 1),
                      (0.25, 'red', 2), (0.30, 'orange', 1),
                      (0.50, 'purple', 1)]:
        ax.axvline(t, color=c, linestyle='--', linewidth=lw,
                   label=f'{int(t*100)}%')
    
    ax.set_xlabel('설비별 처리 비율')
    ax.set_ylabel('설비 수 (log scale)')
    ax.set_title('설비별 처리 비율 분포 (로그 스케일)')
    ax.legend()
    
    # 3-C: 누적 분포 (CDF)
    ax = axes[1, 0]
    sorted_ratios = np.sort(facility_stats['processed_ratio'].dropna().values)
    cumul = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
    ax.plot(sorted_ratios, cumul, linewidth=2)
    
    for t, c in [(0.10, 'green'), (0.20, 'blue'), (0.25, 'red'),
                  (0.30, 'orange'), (0.50, 'purple')]:
        ax.axvline(t, color=c, linestyle='--', linewidth=1)
        if t <= sorted_ratios.max():
            cumul_at_t = (sorted_ratios < t).mean() * 100
            ax.annotate(f'{int(t*100)}%: {cumul_at_t:.2f}%',
                       xy=(t, cumul_at_t), xytext=(t+0.02, cumul_at_t-5),
                       fontsize=9)
    
    ax.set_xlabel('설비별 처리 비율')
    ax.set_ylabel('누적 비율 (%)')
    ax.set_title('처리 비율 누적 분포 (CDF)')
    ax.grid(True, alpha=0.3)
    
    # 3-D: 임계 후보별 제거 설비 비율
    ax = axes[1, 1]
    bars = ax.bar(threshold_eval['threshold_pct'], 
                   threshold_eval['remove_ratio_pct'],
                   edgecolor='black', alpha=0.7, width=3)
    
    # PNNL 25% 강조
    for bar, t in zip(bars, threshold_eval['threshold_pct']):
        if t == 25:
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    ax.set_xlabel('임계치 (%)')
    ax.set_ylabel('제거되는 설비 비율 (%)')
    ax.set_title('임계 후보별 설비 제거 영향 (빨강 = PNNL 25%)')
    
    for i, (t, r) in enumerate(zip(threshold_eval['threshold_pct'],
                                    threshold_eval['remove_ratio_pct'])):
        ax.text(t, r, f'{r:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_distribution_analysis.png"), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # ─────────────────────────────────────
    # (4) 분기점 탐지 (자연스러운 elbow)
    # ─────────────────────────────────────
    print("\n[3] 처리 비율 분포의 분기점 탐지")
    print("\n주요 백분위수:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
        val = facility_stats['processed_ratio'].quantile(q)
        print(f"  {q*100:5.1f}%: {val*100:6.2f}%")
    
    # IQR 기반 outlier 임계
    q1 = facility_stats['processed_ratio'].quantile(0.25)
    q3 = facility_stats['processed_ratio'].quantile(0.75)
    iqr = q3 - q1
    iqr_outlier = q3 + 1.5 * iqr
    
    print(f"\nIQR 기반 outlier 임계: {iqr_outlier*100:.2f}%")
    print(f"  (Q3 + 1.5×IQR = {q3*100:.2f}% + 1.5×{iqr*100:.2f}%)")
    
    # ─────────────────────────────────────
    # (5) 임계 결정 가이드
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[임계 결정 가이드]")
    print("=" * 60)
    
    pnnl_25_remove_pct = (facility_stats['processed_ratio'] >= 0.25).mean() * 100
    print(f"\nPNNL 25% 임계로 제거되는 설비 비율: {pnnl_25_remove_pct:.4f}%")
    
    print("""
판단 기준:
1. 분포에 자연스러운 분기점(elbow)이 있는가?
2. PNNL 25%로 제거되는 설비가 outlier 그룹으로 분리되는가?
3. 분포에 따라 20%, 30% 등 다른 임계가 더 적합할 수 있음
    """)
    
    if pnnl_25_remove_pct > 5:
        print("→ 25%로 너무 많이 제거됨. 임계 상향(30%, 40%) 검토")
    elif pnnl_25_remove_pct < 0.1:
        print("→ 25%로 거의 제거 안 됨. 임계 하향(10%, 15%) 검토 또는 IQR 기반 적용")
    else:
        print("→ 25% 임계 합리적 수준. PNNL 원문 임계 채택 가능")
    
    return facility_stats, threshold_eval


# ============================================================
# 실행
# ============================================================

def main():
    # ─── Phase 1 적용된 데이터 로드 ───
    print("Phase 1 적용 데이터 로딩 중...")
    phase1_path = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\neg\after_phase1.parquet"
    df_phase1 = pd.read_parquet(phase1_path)
    print(f"Phase 1 후 데이터: {df_phase1.shape}\n")
    
    # ─── 원본 데이터 로드 (8단계 처리 비율 계산용) ───
    print("원본 데이터 로딩 중...")
    original_path = r"C:\Users\kjw31\OneDrive\Desktop\Heat-Anomaly-Detection\data\processed\all_data.parquet"
    df_original = pd.read_parquet(original_path)
    print(f"원본 데이터: {df_original.shape}\n")
    
    # ─── Phase 2: 결측 처리 ───
    print("=" * 60)
    print("[Phase 2] 결측 처리 (4~5단계)")
    print("=" * 60 + "\n")
    
    df_phase2 = step4_daily_nan_filter(df_phase1)
    df_phase2 = step5_consecutive_nan_filter(df_phase2)
    
    # ─── Phase 3: 설비 단위 필터 ───
    print("=" * 60)
    print("[Phase 3] 설비 단위 필터 (6~7단계)")
    print("=" * 60 + "\n")
    
    df_phase3 = step6_min_observation_filter(df_phase2)
    df_phase3 = step7_high_missing_facility_filter(df_phase3, df_original)
    
    # 중간 결과 저장
    df_phase3.to_parquet(
        os.path.join(SAVE_DIR, "after_phase3.parquet")
    )
    print(f"Phase 3 적용 후 데이터 저장 완료\n")
    
    # ─── 8단계 임계 검증 ───
    facility_stats, threshold_eval = analyze_facility_processed_ratio(
        df_phase3, df_original, SAVE_DIR
    )
    
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
    print(f"저장 경로: {SAVE_DIR}")
    print("""
저장된 파일:
- 01_facility_processed_ratio_stats.csv: 처리 비율 기본 통계
- 02_facility_processed_ratio.csv: 설비별 처리 비율 상세
- 03_threshold_evaluation.csv: 임계 후보별 영향
- 04_distribution_analysis.png: 시각화 (4개 subplot)
- after_phase3.parquet: Phase 1~3 적용 후 데이터
    """)


if __name__ == "__main__":
    main()
