"""
PNNL-24331 (2015) + Quesada et al. (2024) 기반
열량계 시계열 데이터 전처리 파이프라인 — 최종 적용

8단계 모두 적용 후 최종 데이터 저장
"""

import pandas as pd
import numpy as np
import os
import time

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f"{i}시" for i in range(1, 25)]
FACILITY_COL = "설치"
DATE_COL = "날짜"

# 입력 경로
ORIGINAL_PATH = r"C:\Users\kjw31\OneDrive\Desktop\Heat-Anomaly-Detection\data\processed\all_data.parquet"

# 저장 경로
SAVE_DIR = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\negnan"
os.makedirs(SAVE_DIR, exist_ok=True)

# 임계치 (모든 분석 결과 반영)
OFFSET_WINDOW = 2              # 2단계: 인접 양수 탐색 윈도우
DAILY_NAN_THRESHOLD = 3        # 4단계: 분포 분석 결과 (elbow at 3)
CONSEC_NAN_THRESHOLD = 2       # 5단계: PNNL Option 2 (1시간 초과)
MIN_OBSERVATION_DAYS = 365     # 6단계: Quesada et al. 2024
MAX_MISSING_RATIO = 0.28       # 7단계: 데이터 분포 시각화 분석
MAX_PROCESSED_RATIO = 0.30     # 8단계: 데이터 분포 자연스러운 분기점


# ============================================================
# Phase 1: 음수 처리
# ============================================================

def step1_remove_daily_negative_total(df):
    """
    1단계: 일별 총사용량 음수 행 제거
    근거: PNNL Table 2-2
    """
    daily_sum = df[HOUR_COLS].sum(axis=1, skipna=True)
    n_removed = (daily_sum < 0).sum()
    df_clean = df[daily_sum >= 0].reset_index(drop=True)
    
    print(f"[1단계] 일별 총사용량 음수 행 제거 (PNNL Table 2-2)")
    print(f"        제거: {n_removed:,}행, 잔여: {len(df_clean):,}행\n")
    return df_clean


def step2_offset_negative_processing(df):
    """
    2단계: 인접 양수 상쇄 음수 처리 (PNNL Section 2.4.1)
    """
    df = df.copy()
    data_array = df[HOUR_COLS].values.astype(float)
    
    has_negative = (data_array < 0).any(axis=1)
    rows_with_neg = np.where(has_negative)[0]
    
    counters = {'case1': 0, 'case2': 0, 'case3': 0}
    
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
        
        data_array[row_idx] = values
    
    df[HOUR_COLS] = data_array
    
    print(f"[2단계] 인접 양수 상쇄 처리 (PNNL Section 2.4.1)")
    print(f"        Case 1 (NaN): {counters['case1']:,}, "
          f"Case 2 (보간): {counters['case2']:,}, "
          f"Case 3 (NaN): {counters['case3']:,}\n")
    return df


def step3_no_offset_negative_to_nan(df):
    """
    3단계: 무상쇄 음수 NaN 처리 (PNNL Section 2.4.2)
    """
    df = df.copy()
    n_converted = (df[HOUR_COLS] < 0).sum().sum()
    df[HOUR_COLS] = df[HOUR_COLS].where(~(df[HOUR_COLS] < 0), np.nan)
    
    print(f"[3단계] 무상쇄 음수 NaN 처리 (PNNL Section 2.4.2)")
    print(f"        NaN 변환: {n_converted:,}개\n")
    return df


# ============================================================
# Phase 2: 결측 처리
# ============================================================

def step4_daily_nan_filter(df):
    """
    4단계: 일자 단위 의심 데이터 비율 필터
    조건: 24시간 중 NaN ≥ 3개
    근거: PNNL Section 2.3 + 데이터 분포 분석 (elbow at 3)
    """
    nan_count = df[HOUR_COLS].isna().sum(axis=1)
    n_removed = (nan_count >= DAILY_NAN_THRESHOLD).sum()
    df_clean = df[nan_count < DAILY_NAN_THRESHOLD].reset_index(drop=True)
    
    print(f"[4단계] 일자 단위 NaN ≥ {DAILY_NAN_THRESHOLD}개 제거")
    print(f"        제거: {n_removed:,}일, 잔여: {len(df_clean):,}일\n")
    return df_clean


def step5_consecutive_nan_filter(df):
    """
    5단계: 연속 NaN 길이 평가 (PNNL Section 3 Option 2)
    - 연속 NaN = 1 → 선형 보간
    - 연속 NaN ≥ 2 → 일자 제거
    """
    df = df.copy()
    data_array = df[HOUR_COLS].values.astype(float)
    
    rows_to_remove = []
    rows_to_interpolate = []
    
    for row_idx in range(len(data_array)):
        values = data_array[row_idx]
        is_nan = np.isnan(values)
        
        if not is_nan.any():
            continue
        
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
    n_interpolated = 0
    for row_idx in rows_to_interpolate:
        values = data_array[row_idx]
        interpolated = pd.Series(values).interpolate(
            method='linear', limit_direction='both'
        ).values
        data_array[row_idx] = interpolated
        n_interpolated += 1
    
    df[HOUR_COLS] = data_array
    df_clean = df.drop(index=rows_to_remove).reset_index(drop=True)
    
    print(f"[5단계] 연속 NaN 길이 평가 (PNNL Section 3 Option 2)")
    print(f"        제거 (연속 ≥ {CONSEC_NAN_THRESHOLD}): {len(rows_to_remove):,}일")
    print(f"        보간 (연속 = 1): {n_interpolated:,}일")
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
    
    facility_span = df.groupby(FACILITY_COL)[DATE_COL].agg(['min', 'max'])
    facility_span['days'] = (facility_span['max'] - facility_span['min']).dt.days
    
    facilities_to_keep = facility_span[
        facility_span['days'] >= MIN_OBSERVATION_DAYS
    ].index
    
    n_before = df[FACILITY_COL].nunique()
    df_clean = df[df[FACILITY_COL].isin(facilities_to_keep)].reset_index(drop=True)
    n_after = df_clean[FACILITY_COL].nunique()
    
    print(f"[6단계] 관측 기간 < {MIN_OBSERVATION_DAYS}일 설비 제거 (Quesada 2024)")
    print(f"        제거 설비: {n_before - n_after:,}개")
    print(f"        잔여: {n_after:,}개 설비, {len(df_clean):,}일\n")
    return df_clean


def step7_high_missing_facility_filter(df_processed, df_original):
    """
    7단계: 결측률 ≥ 28% 설비 제거
    근거: 데이터 분포 시각화 분석
    """
    facility_missing = df_original.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].isna().sum().sum() / (len(x) * 24)
    )
    
    facilities_to_keep = facility_missing[
        facility_missing < MAX_MISSING_RATIO
    ].index
    
    n_before = df_processed[FACILITY_COL].nunique()
    df_clean = df_processed[
        df_processed[FACILITY_COL].isin(facilities_to_keep)
    ].reset_index(drop=True)
    n_after = df_clean[FACILITY_COL].nunique()
    
    print(f"[7단계] 결측률 ≥ {MAX_MISSING_RATIO*100:.0f}% 설비 제거")
    print(f"        제거 설비: {n_before - n_after:,}개")
    print(f"        잔여: {n_after:,}개 설비, {len(df_clean):,}일\n")
    return df_clean


def step8_high_processed_ratio_filter(df_processed, df_original):
    """
    8단계: 처리 비율 ≥ 30% 설비 제거 (PNNL Section 2.3 응용)
    근거: PNNL 25% 가이드 + 데이터 분포 자연스러운 분기점 (30~99% 구간 비어있음)
    """
    # 원본 기준 설비별 전체 시점 수
    orig_total_points = df_original.groupby(FACILITY_COL).size() * 24
    
    # 처리 후 살아남은 유효 시점 수
    proc_valid_points = df_processed.groupby(FACILITY_COL).apply(
        lambda x: x[HOUR_COLS].notna().sum().sum()
    )
    
    # 처리 비율 = (원본 - 살아남은 유효) / 원본
    facility_stats = pd.DataFrame({
        'orig_total_points': orig_total_points
    })
    facility_stats['proc_valid_points'] = proc_valid_points
    facility_stats['proc_valid_points'] = facility_stats['proc_valid_points'].fillna(0)
    facility_stats['processed_points'] = (
        facility_stats['orig_total_points'] - facility_stats['proc_valid_points']
    )
    facility_stats['processed_ratio'] = (
        facility_stats['processed_points'] / facility_stats['orig_total_points']
    ).clip(lower=0, upper=1)
    
    # 임계 미만인 설비만 유지
    facilities_to_keep = facility_stats[
        facility_stats['processed_ratio'] < MAX_PROCESSED_RATIO
    ].index
    
    n_before = df_processed[FACILITY_COL].nunique()
    df_clean = df_processed[
        df_processed[FACILITY_COL].isin(facilities_to_keep)
    ].reset_index(drop=True)
    n_after = df_clean[FACILITY_COL].nunique()
    
    print(f"[8단계] 처리 비율 ≥ {MAX_PROCESSED_RATIO*100:.0f}% 설비 제거 (PNNL Section 2.3)")
    print(f"        제거 설비: {n_before - n_after:,}개")
    print(f"        잔여: {n_after:,}개 설비, {len(df_clean):,}일\n")
    
    return df_clean, facility_stats


# ============================================================
# 처리 로그 저장
# ============================================================

def save_processing_log(stats_dict, save_dir):
    """전처리 단계별 로그 저장"""
    log_df = pd.DataFrame(stats_dict)
    log_df.to_csv(
        os.path.join(save_dir, "preprocessing_log.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n로그 저장: {os.path.join(save_dir, 'preprocessing_log.csv')}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    t_start = time.time()
    
    print("=" * 70)
    print("PNNL-24331 (2015) + Quesada et al. (2024) 기반 전처리 파이프라인")
    print("=" * 70 + "\n")
    
    # ─── 데이터 로드 ───
    print("원본 데이터 로딩 중...")
    df_original = pd.read_parquet(ORIGINAL_PATH)
    print(f"원본 데이터: {df_original.shape}")
    print(f"설비 수: {df_original[FACILITY_COL].nunique():,}")
    print(f"기간: {df_original[DATE_COL].min()} ~ {df_original[DATE_COL].max()}\n")
    
    # 처리 로그
    log_data = []
    log_data.append({
        'phase': 'Original',
        'step': 0,
        'description': '원본 데이터',
        'rows': len(df_original),
        'facilities': df_original[FACILITY_COL].nunique()
    })
    
    # ═══════════════════════════════════════
    # Phase 1: 음수 처리
    # ═══════════════════════════════════════
    print("=" * 70)
    print("[Phase 1] 음수 처리 (PNNL Section 2.4)")
    print("=" * 70 + "\n")
    
    df = step1_remove_daily_negative_total(df_original)
    log_data.append({
        'phase': 'Phase 1', 'step': 1,
        'description': '일별 총사용량 음수 행 제거',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    df = step2_offset_negative_processing(df)
    log_data.append({
        'phase': 'Phase 1', 'step': 2,
        'description': '인접 양수 상쇄 처리 (PNNL 2.4.1)',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    df = step3_no_offset_negative_to_nan(df)
    log_data.append({
        'phase': 'Phase 1', 'step': 3,
        'description': '무상쇄 음수 NaN 처리 (PNNL 2.4.2)',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    # 중간 저장 (Phase 1)
    df.to_parquet(os.path.join(SAVE_DIR, "after_phase1.parquet"))
    print(f"중간 저장 (Phase 1): after_phase1.parquet\n")
    
    # ═══════════════════════════════════════
    # Phase 2: 결측 처리
    # ═══════════════════════════════════════
    print("=" * 70)
    print("[Phase 2] 결측 처리 (PNNL Section 3 Option 2)")
    print("=" * 70 + "\n")
    
    df = step4_daily_nan_filter(df)
    log_data.append({
        'phase': 'Phase 2', 'step': 4,
        'description': f'일자 NaN ≥ {DAILY_NAN_THRESHOLD}개 제거',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    df = step5_consecutive_nan_filter(df)
    log_data.append({
        'phase': 'Phase 2', 'step': 5,
        'description': f'연속 NaN ≥ {CONSEC_NAN_THRESHOLD} 제거 + 보간',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    # 중간 저장 (Phase 2)
    df.to_parquet(os.path.join(SAVE_DIR, "after_phase2.parquet"))
    print(f"중간 저장 (Phase 2): after_phase2.parquet\n")
    
    # ═══════════════════════════════════════
    # Phase 3: 설비 단위 품질 필터
    # ═══════════════════════════════════════
    print("=" * 70)
    print("[Phase 3] 설비 단위 품질 필터")
    print("=" * 70 + "\n")
    
    df = step6_min_observation_filter(df)
    log_data.append({
        'phase': 'Phase 3', 'step': 6,
        'description': f'관측 < {MIN_OBSERVATION_DAYS}일 설비 제거 (Quesada 2024)',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    df = step7_high_missing_facility_filter(df, df_original)
    log_data.append({
        'phase': 'Phase 3', 'step': 7,
        'description': f'결측률 ≥ {MAX_MISSING_RATIO*100:.0f}% 설비 제거',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    df, facility_stats = step8_high_processed_ratio_filter(df, df_original)
    log_data.append({
        'phase': 'Phase 3', 'step': 8,
        'description': f'처리 비율 ≥ {MAX_PROCESSED_RATIO*100:.0f}% 설비 제거 (PNNL 2.3)',
        'rows': len(df),
        'facilities': df[FACILITY_COL].nunique()
    })
    
    # ═══════════════════════════════════════
    # 최종 저장
    # ═══════════════════════════════════════
    print("=" * 70)
    print("[최종 결과 저장]")
    print("=" * 70 + "\n")
    
    final_path = os.path.join(SAVE_DIR, "final_preprocessed.parquet")
    df.to_parquet(final_path)
    print(f"최종 데이터 저장: {final_path}")
    print(f"  최종 shape: {df.shape}")
    print(f"  최종 설비 수: {df[FACILITY_COL].nunique():,}")
    
    # 처리 로그 저장
    save_processing_log(log_data, SAVE_DIR)
    
    # 8단계 처리 비율 통계 저장 (참고용)
    facility_stats.to_csv(
        os.path.join(SAVE_DIR, "facility_processed_ratio_final.csv"),
        encoding="utf-8-sig"
    )
    
    # ═══════════════════════════════════════
    # 처리 요약
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("[전처리 결과 요약]")
    print("=" * 70)
    
    summary_df = pd.DataFrame(log_data)
    
    # 단계별 변화량 계산
    summary_df['rows_diff'] = summary_df['rows'].diff().fillna(0).astype(int)
    summary_df['facilities_diff'] = summary_df['facilities'].diff().fillna(0).astype(int)
    
    print("\n단계별 변화:")
    print(summary_df.to_string(index=False))
    
    # 전체 영향
    orig_rows = log_data[0]['rows']
    final_rows = log_data[-1]['rows']
    orig_facilities = log_data[0]['facilities']
    final_facilities = log_data[-1]['facilities']
    
    print(f"\n전체 데이터 변화:")
    print(f"  행: {orig_rows:,} → {final_rows:,} "
          f"({(final_rows/orig_rows*100):.2f}% 유지, "
          f"{((1-final_rows/orig_rows)*100):.2f}% 제거)")
    print(f"  설비: {orig_facilities:,} → {final_facilities:,} "
          f"({(final_facilities/orig_facilities*100):.2f}% 유지, "
          f"{((1-final_facilities/orig_facilities)*100):.2f}% 제거)")
    
    print(f"\n총 소요 시간: {time.time() - t_start:.1f}초")
    print(f"\n저장 위치: {SAVE_DIR}")
    print("""
저장된 파일:
- after_phase1.parquet: Phase 1 (1~3단계) 적용 후 중간 결과
- after_phase2.parquet: Phase 2 (4~5단계) 적용 후 중간 결과
- final_preprocessed.parquet: ★ 최종 전처리 완료 데이터 ★
- preprocessing_log.csv: 단계별 처리 로그
- facility_processed_ratio_final.csv: 설비별 처리 비율 통계
    """)


if __name__ == "__main__":
    main()