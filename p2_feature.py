"""
이상 탐지 모델을 위한 피처 엔지니어링 + 상관관계 매트릭스 분석
- 16개 수치형 + 1개 카테고리(season) 피처
- One-hot encoding 후 Pearson/Spearman 상관관계 + GVIF 다중공선성 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ============================================================
# 0. 설정
# ============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INPUT_PATH = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\negnan\final_preprocessed.parquet"
SAVE_DIR = r"C:\Users\kjw31\OneDrive\Desktop\heatdata\features"
os.makedirs(SAVE_DIR, exist_ok=True)

HOUR_COLS = [f"{i}시" for i in range(1, 25)]
FACILITY_COL = "설치"
DATE_COL = "날짜"

# 한국 공휴일 (2021-2025)
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


# ============================================================
# 1. 피처 엔지니어링
# ============================================================

def engineer_features(df):
    """
    24시간 시계열 → 이상 탐지용 피처 생성 (벡터화)
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    hour_data = df[HOUR_COLS].values.astype(float)
    n_rows = len(hour_data)
    
    print(f"[피처 생성 시작] {n_rows:,}행 처리\n")
    
    # ─────────────────────────────────────
    # [1/5] 기본 통계
    # ─────────────────────────────────────
    print("  [1/5] 기본 통계 피처...")
    
    df['daily_mean'] = np.nanmean(hour_data, axis=1)
    daily_std = np.nanstd(hour_data, axis=1)
    df['hourly_std'] = daily_std
    df['cv'] = np.where(
        df['daily_mean'] > 0, daily_std / df['daily_mean'], np.nan
    )
    
    # 피크 시간 (벡터화)
    all_nan_mask = np.all(np.isnan(hour_data), axis=1)
    safe_data_filled = np.where(np.isnan(hour_data), -np.inf, hour_data)
    peak_hour = np.argmax(safe_data_filled, axis=1) + 1
    peak_hour = np.where(all_nan_mask, np.nan, peak_hour)
    df['peak_hour'] = peak_hour
    
    df['baseload'] = np.nanmin(hour_data, axis=1)
    
    # ─────────────────────────────────────
    # [2/5] 패턴 비율
    # ─────────────────────────────────────
    print("  [2/5] 패턴 비율 피처...")
    
    daily_sum = np.nansum(hour_data, axis=1)
    daily_sum_safe = np.where(daily_sum > 0, daily_sum, np.nan)
    
    # 시간대별 비율
    hour_ratios = hour_data / daily_sum_safe[:, None]
    hour_ratio_df = pd.DataFrame(
        hour_ratios,
        columns=[f'hour_ratio_{h}' for h in range(1, 25)],
        index=df.index
    )
    df = pd.concat([df, hour_ratio_df], axis=1)
    
    # 야간 비율 (22-06시)
    night_idx = [21, 22, 23, 0, 1, 2, 3, 4, 5]
    night_sum = np.nansum(hour_data[:, night_idx], axis=1)
    df['night_ratio'] = night_sum / daily_sum_safe
    
    # 주간 비율 (09-18시)
    day_idx = list(range(8, 18))
    day_sum = np.nansum(hour_data[:, day_idx], axis=1)
    df['day_ratio'] = day_sum / daily_sum_safe
    
    # ─────────────────────────────────────
    # [3/5] 추세 비교
    # ─────────────────────────────────────
    print("  [3/5] 추세 비교 피처...")
    
    df = df.sort_values([FACILITY_COL, DATE_COL]).reset_index(drop=True)
    
    ma7 = df.groupby(FACILITY_COL, sort=False)['daily_mean'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['ma7_ratio'] = np.where(ma7 > 0, df['daily_mean'] / ma7, np.nan)
    
    ma30 = df.groupby(FACILITY_COL, sort=False)['daily_mean'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    df['ma30_ratio'] = np.where(ma30 > 0, df['daily_mean'] / ma30, np.nan)
    
    # ─────────────────────────────────────
    # [4/5] 시계열 패턴
    # ─────────────────────────────────────
    print("  [4/5] 시계열 패턴 피처...")
    
    # Lag-1 자기상관 (벡터화)
    x = hour_data[:, :-1]
    y = hour_data[:, 1:]
    valid = ~(np.isnan(x) | np.isnan(y))
    n_valid = valid.sum(axis=1)
    
    x_masked = np.where(valid, x, 0)
    y_masked = np.where(valid, y, 0)
    sum_x = x_masked.sum(axis=1)
    sum_y = y_masked.sum(axis=1)
    
    mean_x = np.where(n_valid > 0, sum_x / n_valid, np.nan)
    mean_y = np.where(n_valid > 0, sum_y / n_valid, np.nan)
    
    dx = np.where(valid, x - mean_x[:, None], 0)
    dy = np.where(valid, y - mean_y[:, None], 0)
    
    cov_xy = (dx * dy).sum(axis=1)
    var_x = (dx * dx).sum(axis=1)
    var_y = (dy * dy).sum(axis=1)
    
    denom = np.sqrt(var_x * var_y)
    autocorr = np.where(
        (denom > 0) & (n_valid >= 3), cov_xy / denom, np.nan
    )
    df['autocorr_lag1'] = autocorr
    
    # 시간 변화율
    hour_diff = np.diff(hour_data, axis=1)
    df['hour_diff_max'] = np.nanmax(np.abs(hour_diff), axis=1)
    df['hour_diff_std'] = np.nanstd(hour_diff, axis=1)
    
    # 0의 개수
    df['zero_count'] = np.sum(hour_data == 0, axis=1)
    
    # ─────────────────────────────────────
    # [5/5] 메타 정보
    # ─────────────────────────────────────
    print("  [5/5] 메타 정보 피처...")
    
    month = df[DATE_COL].dt.month
    
    season_map = {
        12: '겨울', 1: '겨울', 2: '겨울',
        3: '봄', 4: '봄', 5: '봄',
        6: '여름', 7: '여름', 8: '여름',
        9: '가을', 10: '가을', 11: '가을'
    }
    df['season'] = month.map(season_map)
    df['is_heating_season'] = month.isin([11, 12, 1, 2, 3]).astype(int)
    
    date_normalized = df[DATE_COL].dt.normalize()
    df['is_holiday'] = date_normalized.isin(KOREA_HOLIDAYS).astype(int)
    df['is_weekend'] = (df[DATE_COL].dt.dayofweek >= 5).astype(int)
    
    print(f"\n[피처 생성 완료] 총 {len(df.columns)}개 컬럼")
    return df


# ============================================================
# 2. GVIF 계산 (별도 함수)
# ============================================================

def compute_gvif(df_data, variable_groups):
    """
    Generalized VIF 계산 (Fox & Monette, 1992)
    
    Parameters:
    -----------
    df_data : pd.DataFrame
        분석할 데이터 (표준화 권장)
    variable_groups : dict
        {그룹명: [컬럼명 리스트]}
    
    Returns:
    --------
    pd.DataFrame: 그룹별 GVIF 결과
    """
    results = []
    
    R_full = df_data.corr().values
    det_R_full = np.linalg.det(R_full)
    
    if abs(det_R_full) < 1e-12:
        print("  ⚠ 전체 상관 행렬의 determinant가 0에 가까움 — 완전 다중공선성 의심")
    
    for group_name, cols in variable_groups.items():
        x_cols = cols
        other_cols = [c for c in df_data.columns if c not in x_cols]
        
        if len(other_cols) == 0:
            continue
        
        det_R_xx = (
            np.linalg.det(df_data[x_cols].corr().values) 
            if len(x_cols) > 1 else 1.0
        )
        det_R_zz = (
            np.linalg.det(df_data[other_cols].corr().values) 
            if len(other_cols) > 1 else 1.0
        )
        
        gvif = (det_R_xx * det_R_zz) / det_R_full
        df_group = len(x_cols)
        gvif_adjusted = gvif ** (1 / (2 * df_group))
        
        results.append({
            'group': group_name,
            'n_variables': df_group,
            'GVIF': round(gvif, 3),
            'GVIF^(1/(2·df))': round(gvif_adjusted, 3),
            'severity': (
                '심각 (>3.16)' if gvif_adjusted > 3.16 
                else '주의 (>2.24)' if gvif_adjusted > 2.24
                else '양호'
            )
        })
    
    return pd.DataFrame(results).sort_values(
        'GVIF^(1/(2·df))', ascending=False
    )


# ============================================================
# 3. 상관관계 매트릭스 분석
# ============================================================

def analyze_correlation_matrix(df, save_dir):
    """
    카테고리 변수 One-hot encoding 후 상관관계 + GVIF 분석
    """
    print("\n" + "=" * 60)
    print("[상관관계 매트릭스 분석]")
    print("=" * 60)
    
    # ─────────────────────────────────────
    # 분석 대상 피처 정의
    # ─────────────────────────────────────
    numeric_features = [
        'daily_mean', 'cv', 'hourly_std', 'peak_hour', 'baseload',
        'night_ratio', 'day_ratio',
        'ma7_ratio', 'ma30_ratio',
        'autocorr_lag1', 'hour_diff_max', 'hour_diff_std',
        'zero_count',
        'is_heating_season', 'is_holiday', 'is_weekend'
    ]
    categorical_features = ['season']
    
    # ─────────────────────────────────────
    # One-hot encoding
    # ─────────────────────────────────────
    print("\n카테고리 변수 One-hot encoding...")
    
    df_for_corr = df[numeric_features + categorical_features].copy()
    df_encoded = pd.get_dummies(
        df_for_corr,
        columns=categorical_features,
        prefix=categorical_features,
        drop_first=False
    )
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
    
    print(f"인코딩 후 컬럼 수: {len(df_encoded.columns)}")
    
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
    n_before = len(df_encoded)
    df_clean = df_encoded.dropna()
    n_after = len(df_clean)
    print(f"결측 제거: {n_before:,} → {n_after:,} 행\n")
    
    # ─────────────────────────────────────
    # (1) Pearson / Spearman
    # ─────────────────────────────────────
    print("[1] Pearson / Spearman 상관계수 계산 중...")
    corr_pearson = df_clean.corr(method='pearson')
    corr_spearman = df_clean.corr(method='spearman')
    
    corr_pearson.to_csv(
        os.path.join(save_dir, "01_correlation_pearson.csv"),
        encoding="utf-8-sig"
    )
    corr_spearman.to_csv(
        os.path.join(save_dir, "02_correlation_spearman.csv"),
        encoding="utf-8-sig"
    )
    
    # ─────────────────────────────────────
    # (2) 시각화
    # ─────────────────────────────────────
    print("[2] 상관관계 히트맵 생성 중...")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    sns.heatmap(
        corr_pearson, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        cbar_kws={'shrink': 0.8}, ax=axes[0],
        annot_kws={'size': 8}
    )
    axes[0].set_title('Pearson 상관계수 (선형 관계)', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.heatmap(
        corr_spearman, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        cbar_kws={'shrink': 0.8}, ax=axes[1],
        annot_kws={'size': 8}
    )
    axes[1].set_title('Spearman 상관계수 (단조 관계)', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "03_correlation_heatmap.png"),
        dpi=100, bbox_inches='tight'
    )
    plt.close()
    
    # ─────────────────────────────────────
    # (3) 강한 상관관계 추출
    # ─────────────────────────────────────
    print("\n[3] 강한 상관관계 (|r| ≥ 0.7) 추출")
    
    high_corr_pairs = []
    for i in range(len(corr_pearson.columns)):
        for j in range(i+1, len(corr_pearson.columns)):
            r = corr_pearson.iloc[i, j]
            if abs(r) >= 0.7:
                high_corr_pairs.append({
                    'feature1': corr_pearson.columns[i],
                    'feature2': corr_pearson.columns[j],
                    'pearson': round(r, 3),
                    'spearman': round(corr_spearman.iloc[i, j], 3)
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values(
            'pearson', key=lambda x: x.abs(), ascending=False
        )
        print(high_corr_df.to_string(index=False))
        high_corr_df.to_csv(
            os.path.join(save_dir, "04_high_correlation_pairs.csv"),
            index=False, encoding="utf-8-sig"
        )
    else:
        print("강한 상관관계 없음 (|r| < 0.7)")
    
    # ─────────────────────────────────────
    # (4) 시간대 비율 24개 별도 시각화
    # ─────────────────────────────────────
    print("\n[4] 시간대 비율 24개 상관관계 분석")
    
    hour_ratio_cols = [f'hour_ratio_{h}' for h in range(1, 25)]
    df_hour = df[hour_ratio_cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr_hour = df_hour.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_hour, annot=False, cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        cbar_kws={'shrink': 0.8}, ax=ax
    )
    ax.set_title('시간대 비율(1~24시) 상관관계', fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "05_hour_ratio_correlation.png"),
        dpi=100, bbox_inches='tight'
    )
    plt.close()
    
    # ─────────────────────────────────────
    # (5) GVIF 분석
    # ─────────────────────────────────────
    print("\n[5] GVIF 분석 (카테고리 변수 그룹 단위 다중공선성)")
    print("(GVIF^(1/(2·df)) > √10 ≈ 3.16: 심각, > √5 ≈ 2.24: 주의)")
    
    try:
        # 변수 그룹 정의
        season_cols = [c for c in df_clean.columns if c.startswith('season_')]
        
        variable_groups = {
            'daily_mean': ['daily_mean'],
            'cv': ['cv'],
            'hourly_std': ['hourly_std'],
            'peak_hour': ['peak_hour'],
            'baseload': ['baseload'],
            'night_ratio': ['night_ratio'],
            'day_ratio': ['day_ratio'],
            'ma7_ratio': ['ma7_ratio'],
            'ma30_ratio': ['ma30_ratio'],
            'autocorr_lag1': ['autocorr_lag1'],
            'hour_diff_max': ['hour_diff_max'],
            'hour_diff_std': ['hour_diff_std'],
            'zero_count': ['zero_count'],
            'is_heating_season': ['is_heating_season'],
            'is_holiday': ['is_holiday'],
            'is_weekend': ['is_weekend'],
            'season': season_cols,
        }
        
        # 표준화
        df_std = (df_clean - df_clean.mean()) / df_clean.std()
        df_std = df_std.dropna()
        
        # 상수 분산 컬럼 제거
        non_constant_cols = [c for c in df_std.columns if df_std[c].std() > 0]
        df_std = df_std[non_constant_cols]
        
        # 변수 그룹 업데이트
        variable_groups = {
            k: [c for c in v if c in df_std.columns]
            for k, v in variable_groups.items()
        }
        variable_groups = {k: v for k, v in variable_groups.items() if v}
        
        # GVIF 계산
        gvif_result = compute_gvif(df_std, variable_groups)
        print("\n[GVIF 결과]")
        print(gvif_result.to_string(index=False))
        
        gvif_result.to_csv(
            os.path.join(save_dir, "06_gvif_analysis.csv"),
            index=False, encoding="utf-8-sig"
        )
        
        # 비교용: 일반 VIF
        print("\n[참고] 일반 VIF (카테고리도 개별 더미로 계산)")
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        df_vif_const = add_constant(df_std)
        vif_data = pd.DataFrame()
        vif_data['feature'] = df_std.columns
        vif_data['VIF'] = [
            variance_inflation_factor(df_vif_const.values, i+1)
            for i in range(len(df_std.columns))
        ]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        print(vif_data.to_string(index=False))
        
        vif_data.to_csv(
            os.path.join(save_dir, "06b_regular_vif_for_comparison.csv"),
            index=False, encoding="utf-8-sig"
        )
        
    except ImportError:
        print("statsmodels 미설치. pip install statsmodels")
    except Exception as e:
        print(f"GVIF 계산 오류: {e}")
        import traceback
        traceback.print_exc()
    
    # ─────────────────────────────────────
    # (6) 피처 통계
    # ─────────────────────────────────────
    print("\n[6] 피처 기본 통계")
    feature_stats = df_clean.describe()
    print(feature_stats.T[['mean', 'std', 'min', '50%', 'max']])
    
    feature_stats.to_csv(
        os.path.join(save_dir, "07_feature_statistics.csv"),
        encoding="utf-8-sig"
    )
    
    return corr_pearson, corr_spearman


# ============================================================
# 4. 메인 실행
# ============================================================

def main():
    t_start = time.time()
    
    print("데이터 로딩 중...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"입력 데이터: {df.shape}")
    print(f"컬럼: {list(df.columns)}... (총 {len(df.columns)}개)")
    print(f"설비 수: {df[FACILITY_COL].nunique():,}")
    print(f"기간: {df[DATE_COL].min()} ~ {df[DATE_COL].max()}\n")
    
    # 피처 엔지니어링
    df_features = engineer_features(df)
    
    # 결과 저장
    df_features.to_parquet(
        os.path.join(SAVE_DIR, "features.parquet")
    )
    print(f"\n피처 데이터 저장: {os.path.join(SAVE_DIR, 'features.parquet')}")
    print(f"피처 데이터 shape: {df_features.shape}")
    
    # 상관관계 분석
    t2 = time.time()
    corr_pearson, corr_spearman = analyze_correlation_matrix(
        df_features, SAVE_DIR
    )
    print(f"\n상관관계 분석 소요 시간: {time.time() - t2:.1f}초")
    
    print("\n" + "=" * 60)
    print(f"전체 완료 — 총 소요 시간: {time.time() - t_start:.1f}초")
    print("=" * 60)
    print(f"저장 경로: {SAVE_DIR}")
    print("""
저장된 파일:
- features.parquet: 피처 엔지니어링 완료 데이터
- 01_correlation_pearson.csv: Pearson 상관계수
- 02_correlation_spearman.csv: Spearman 상관계수
- 03_correlation_heatmap.png: 상관관계 시각화
- 04_high_correlation_pairs.csv: 강한 상관관계 쌍
- 05_hour_ratio_correlation.png: 시간대 비율 상관관계
- 06_gvif_analysis.csv: GVIF 분석 (카테고리 그룹 단위)
- 06b_regular_vif_for_comparison.csv: 일반 VIF (참고)
- 07_feature_statistics.csv: 피처 기본 통계
    """)


if __name__ == "__main__":
    main()