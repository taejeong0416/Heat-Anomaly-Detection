"""
Phase 6. 이상 유형 분류 및 결과 해석
Phase 5에서 이상으로 판정된 샘플(`is_anomaly==1`)에 대해
이상유형정의.md / 분석플랜 5-1의 7개 이상사용 유형 규칙을 적용하고,
SHAP 해석 + 결과 시각화를 수행한다.

이상사용 유형 (분석플랜 5-1):
  1) 급증형         : 전일 대비 변화율(ma7_ratio) 상위 + context_zscore 상위
  2) 야간이상형      : night_ratio가 종별·월 그룹 평균 + 2σ 이상
  3) 패턴이탈형      : gmm_log_likelihood 하위 N% OR ae_score 상위 N%
  4) 장기미사용후급증형 : 활성시즌 내 zero_count ≥ N + ma7_ratio 상위
  5) 계절역행형      : 비활성시즌 사용량 상위 N%
  6) 주말이상형      : 업무용/공공용 & 주말·공휴일 & 사용량이 평일 수준 이상
  7) 기저유량이상형  : baseload 종별·시즌 그룹 상위 N%

냉수용은 활성시즌 정의를 반전(냉방시즌=활성).

사용법:
  python scripts/phase6_classification.py
"""

import pandas as pd
import numpy as np
import json
import os
import gc
import time
import pickle

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f'{i}시' for i in range(1, 25)]
HOUR_RATIO_COLS = [f'hour_ratio_{i}' for i in range(1, 25)]
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED_DIR = 'data/processed'
FIG_DIR = 'outputs/figures'
MODEL_DIR = 'outputs/phase5_models'
PHASE6_DIR = 'outputs/phase6'

TYPES = ['주택용', '업무용', '공공용', '냉수용']

# Phase 5와 동일한 IF 입력 피처 (SHAP용)
IF_FEATURES = (
    HOUR_RATIO_COLS
    + [
        '총사용량',
        'cv', 'hourly_std', 'peak_hour', 'baseload',
        'night_ratio', 'day_ratio',
        'ma7_ratio', 'ma30_ratio',
        'autocorr_lag1', 'hour_diff_max', 'hour_diff_std',
        'zero_count',
        'gmm_log_likelihood', 'context_zscore',
    ]
)

# 유형 판정 임계 (이상 샘플 내부 상대 기준)
SURGE_MA7_QTILE = 0.90         # 급증형: ma7_ratio 상위 10%
SURGE_Z_QTILE = 0.90           # 급증형: context_zscore 상위 10%
NIGHT_SIGMA = 2.0              # 야간: 종별·월 평균 + 2σ
PATTERN_LL_QTILE = 0.10        # 패턴이탈: gmm_log_likelihood 하위 10%
PATTERN_AE_QTILE = 0.90        # 패턴이탈: ae_score 상위 10%
LONG_ZERO_DAYS = 7             # 장기미사용 임계 zero_count
LONG_MA7_QTILE = 0.80          # 장기미사용 후 급증: ma7_ratio 상위 20%
SEASON_REV_QTILE = 0.95        # 계절역행: 비활성시즌 사용량 상위 5%
WEEKEND_RATIO = 0.8            # 주말·공휴일이 평일 수준 80% 이상
BASELOAD_QTILE = 0.95          # 기저유량 상위 5%

REPRESENTATIVE_N = 5           # 유형별 대표 사례 수
SHAP_SAMPLE = 1000             # SHAP 계산 샘플 (종별별)
RANDOM_STATE = 42

log = []


def log_step(msg):
    log.append(msg)
    print(f'  → {msg}')


# ============================================================
# 1. 유형 분류 규칙
# ============================================================

def add_active_season(df):
    """종별에 따라 활성시즌 컬럼 추가 (냉수용은 냉방시즌 반전 적용)."""
    is_cold = df['종별'] == '냉수용'
    df['활성시즌'] = np.where(
        is_cold, df['냉방시즌'].astype(bool), df['난방시즌'].astype(bool)
    )
    return df


def classify_types(df_anom, df_all):
    """
    이상 샘플(df_anom)에 7개 유형 규칙을 적용.

    Parameters:
      df_anom: is_anomaly==1 행 (분류 대상)
      df_all : 전체 데이터 (그룹 통계 산출용)

    Returns:
      df_anom: 'type_*' boolean 컬럼 7개 + 'primary_type' + 'secondary_types'
    """
    print('\n[유형 규칙 적용]')

    # ── (그룹 통계: 종별·월 night_ratio 평균/표준편차) ──
    nr_stats = (
        df_all.groupby(['종별', '월'], observed=True)['night_ratio']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'nr_mean', 'std': 'nr_std'})
    )
    df_anom = df_anom.merge(nr_stats, on=['종별', '월'], how='left')

    # ── (그룹 통계: 종별·활성시즌 baseload 95%분위) ──
    bl_q = (
        df_all.groupby(['종별', '활성시즌'], observed=True)['baseload']
        .quantile(BASELOAD_QTILE).reset_index()
        .rename(columns={'baseload': 'bl_thr'})
    )
    df_anom = df_anom.merge(bl_q, on=['종별', '활성시즌'], how='left')

    # ── (그룹 통계: 종별·활성시즌 사용량 95%분위 — 계절역행용) ──
    use_q = (
        df_all.groupby(['종별', '활성시즌'], observed=True)['총사용량']
        .quantile(SEASON_REV_QTILE).reset_index()
        .rename(columns={'총사용량': 'use_q95'})
    )
    df_anom = df_anom.merge(use_q, on=['종별', '활성시즌'], how='left')

    # ── (이상 샘플 내부 분위 — 급증/패턴이탈) ──
    q_ma7 = df_anom['ma7_ratio'].quantile(SURGE_MA7_QTILE)
    q_z = df_anom['context_zscore'].abs().quantile(SURGE_Z_QTILE)
    q_ll = df_anom['gmm_log_likelihood'].quantile(PATTERN_LL_QTILE)
    q_ae = df_anom['ae_score'].quantile(PATTERN_AE_QTILE)
    q_long_ma7 = df_anom['ma7_ratio'].quantile(LONG_MA7_QTILE)

    log_step(f'임계: ma7≥{q_ma7:.3f}, |z|≥{q_z:.2f}, '
             f'logLL≤{q_ll:.2f}, ae≥{q_ae:.4f}')

    # ── 7개 유형 라벨 ──
    df_anom['type_급증형'] = (
        (df_anom['ma7_ratio'] >= q_ma7)
        & (df_anom['context_zscore'].abs() >= q_z)
    )

    night_thr = df_anom['nr_mean'] + NIGHT_SIGMA * df_anom['nr_std']
    df_anom['type_야간이상형'] = df_anom['night_ratio'] >= night_thr

    df_anom['type_패턴이탈형'] = (
        (df_anom['gmm_log_likelihood'] <= q_ll)
        | (df_anom['ae_score'] >= q_ae)
    )

    df_anom['type_장기미사용후급증형'] = (
        (df_anom['활성시즌'] == True)
        & (df_anom['zero_count'] >= LONG_ZERO_DAYS)
        & (df_anom['ma7_ratio'] >= q_long_ma7)
    )

    df_anom['type_계절역행형'] = (
        (df_anom['활성시즌'] == False)
        & (df_anom['총사용량'] >= df_anom['use_q95'])
    )

    is_office_pub = df_anom['종별'].isin(['업무용', '공공용'])
    is_off = (
        (df_anom['is_weekend'].astype(bool))
        | (df_anom['is_holiday'].astype(bool))
    )
    df_anom['type_주말이상형'] = (
        is_office_pub & is_off
        & (df_anom['총사용량'] >= df_anom['use_q95'] * WEEKEND_RATIO)
    )

    df_anom['type_기저유량이상형'] = df_anom['baseload'] >= df_anom['bl_thr']

    type_cols = [c for c in df_anom.columns if c.startswith('type_')]

    # ── 주 유형/보조 유형 ──
    # 우선순위: 패턴이탈 > 급증 > 야간 > 기저 > 장기→급증 > 계절역행 > 주말
    PRIORITY = [
        'type_패턴이탈형',
        'type_급증형',
        'type_야간이상형',
        'type_기저유량이상형',
        'type_장기미사용후급증형',
        'type_계절역행형',
        'type_주말이상형',
    ]

    def pick_primary(row):
        for c in PRIORITY:
            if row.get(c, False):
                return c.replace('type_', '')
        return '미분류'

    df_anom['primary_type'] = df_anom[type_cols].apply(pick_primary, axis=1)

    def pick_secondary(row):
        prim = 'type_' + row['primary_type']
        return ','.join(
            c.replace('type_', '')
            for c in type_cols if row.get(c, False) and c != prim
        )

    df_anom['secondary_types'] = df_anom.apply(pick_secondary, axis=1)

    # ── 심각도 (0~1로 정규화된 IF score) ──
    if_min = df_anom['if_score'].min()
    if_max = df_anom['if_score'].max()
    if if_max > if_min:
        df_anom['severity'] = (
            (df_anom['if_score'] - if_min) / (if_max - if_min)
        ).astype(np.float32)
    else:
        df_anom['severity'] = np.float32(0.5)

    # 통계 출력
    counts = df_anom['primary_type'].value_counts()
    print('\n  주 유형 분포:')
    for k, v in counts.items():
        print(f'    {k}: {v:,} ({v/len(df_anom)*100:.1f}%)')

    return df_anom


# ============================================================
# 2. SHAP 해석
# ============================================================

def run_shap(df_all, type_dist_idx):
    """
    Phase 5 IF 모델에 SHAP 적용.
    종별별로 (a) summary bar + (b) 유형별 대표 사례 waterfall 저장.
    """
    print('\n[SHAP 해석]')

    try:
        import shap
    except ImportError:
        log_step('SHAP 미설치 → 건너뜀 (pip install shap 필요)')
        return {}

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    shap_summary = {}
    for type_name in TYPES:
        model_path = os.path.join(MODEL_DIR, f'if_{type_name}.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'if_scaler_{type_name}.pkl')
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f'  {type_name}: 모델 없음 → skip')
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        mask = (df_all['종별'] == type_name) & (df_all['is_anomaly'] == 1)
        if mask.sum() < 10:
            continue

        sub = df_all.loc[mask]
        sample = sub.sample(
            n=min(SHAP_SAMPLE, len(sub)), random_state=RANDOM_STATE
        )
        X_raw = sample[IF_FEATURES].fillna(0).values.astype(np.float32)
        X = scaler.transform(X_raw).astype(np.float32)

        print(f'  {type_name}: SHAP 계산 ({len(X):,}행)')
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
        except Exception as e:
            print(f'    TreeExplainer 실패 ({e}), KernelExplainer 미시도')
            continue

        # (a) summary bar
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals, X, feature_names=IF_FEATURES,
            plot_type='bar', show=False, max_display=15,
        )
        plt.title(f'{type_name} | IF SHAP feature importance')
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f'phase06_shap_summary_{type_name}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        # 평균 |shap| 상위 5개 기록
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:10]
        shap_summary[type_name] = [
            {'feature': IF_FEATURES[i], 'mean_abs_shap': float(mean_abs[i])}
            for i in order
        ]

        # (b) 유형별 대표 1건 waterfall
        for ptype, rep_idx in type_dist_idx.get(type_name, {}).items():
            if rep_idx is None or rep_idx not in sample.index:
                # 인덱스 미일치 → 패스 (대표 사례가 sample에 없을 수 있음)
                continue
            pos = sample.index.get_loc(rep_idx)
            try:
                expl = shap.Explanation(
                    values=shap_vals[pos],
                    base_values=explainer.expected_value
                    if np.isscalar(explainer.expected_value)
                    else explainer.expected_value[0],
                    data=X[pos],
                    feature_names=IF_FEATURES,
                )
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(expl, show=False, max_display=12)
                plt.title(f'{type_name} | {ptype} 대표 사례')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        FIG_DIR,
                        f'phase06_shap_waterfall_{type_name}_{ptype}.png',
                    ),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()
            except Exception as e:
                print(f'    waterfall 실패 ({type_name}/{ptype}): {e}')

        del model, scaler, shap_vals, X, X_raw
        gc.collect()

    return shap_summary


# ============================================================
# 3. 결과 시각화
# ============================================================

def plot_type_pie(df_anom, save_path):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    counts = df_anom['primary_type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values, labels=counts.index,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10},
    )
    ax.set_title('이상 유형 분포 (주 유형)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_monthly_trend(df_anom, save_path):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    df_anom = df_anom.copy()
    df_anom['연월'] = pd.to_datetime(df_anom[DATE_COL]).dt.to_period('M').astype(str)
    monthly = (
        df_anom.groupby(['연월', 'primary_type'], observed=True)
        .size().unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly.plot(ax=ax, marker='o', markersize=3, linewidth=1.2)
    ax.set_title('월별 이상 건수 추이 (유형별)')
    ax.set_xlabel('연월')
    ax.set_ylabel('건수')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_branch_heatmap(df_anom, save_path):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    pivot = (
        df_anom.groupby(['지사', '종별'], observed=True)
        .size().unstack(fill_value=0)
    )
    # 행 정렬: 총 이상 건수 내림차순, 상위 30개
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(30)

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title('지사·종별 이상 빈도 (상위 30 지사)')
    fig.colorbar(im, ax=ax, label='이상 건수')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_representative_cases(df_anom, df_all, save_dir):
    """유형별 대표 사례의 24시간 사용량 곡선 + 정상 패턴 오버레이."""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    rep_index = {t: {} for t in TYPES}

    for ptype, sub in df_anom.groupby('primary_type', observed=True):
        if ptype == '미분류':
            continue
        # 심각도 상위 N건
        top = sub.nlargest(REPRESENTATIVE_N, 'severity')
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            type_name = row['종별']
            # 정상 평균 곡선 (종별·활성시즌)
            mask_norm = (
                (df_all['종별'] == type_name)
                & (df_all['활성시즌'] == row['활성시즌'])
                & (df_all['is_anomaly'] == 0)
            )
            normal_mean = df_all.loc[mask_norm, HOUR_COLS].mean()

            fig, ax = plt.subplots(figsize=(11, 5))
            hours = list(range(1, 25))
            ax.plot(hours, row[HOUR_COLS].values, 'o-',
                    color='#E53935', linewidth=2,
                    label=f'이상 ({row[DATE_COL]})')
            ax.plot(hours, normal_mean.values, '--',
                    color='#1E88E5', linewidth=1.5,
                    label='정상 평균')
            ax.set_xlabel('시간')
            ax.set_ylabel('사용량 (Gcal)')
            ax.set_title(
                f'{ptype} | {type_name} | {row[FACILITY_COL]} '
                f'(severity={row["severity"]:.2f})'
            )
            ax.legend()
            ax.set_xticks(hours)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f'phase06_case_{ptype}_{rank}.png'
                ),
                dpi=150, bbox_inches='tight'
            )
            plt.close()

            # SHAP 매핑용 대표 인덱스 (rank=1)만 저장
            if rank == 1 and type_name in rep_index:
                rep_index[type_name][ptype] = row.name

    return rep_index


# ============================================================
# 4. 메인 파이프라인
# ============================================================

def run():
    t_start = time.time()

    print('=' * 60)
    print('Phase 6. 이상 유형 분류 및 결과 해석')
    print('=' * 60)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PHASE6_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    input_path = f'{PROCESSED_DIR}/features_phase5.parquet'

    import pyarrow.parquet as pq
    _table = pq.read_table(input_path)
    df_all = _table.to_pandas(self_destruct=True)
    del _table
    gc.collect()

    df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL])
    df_all = add_active_season(df_all)
    print(f'  전체: {len(df_all):,}행, {len(df_all.columns)}개 컬럼')

    # ── 이상 샘플 분리 ──
    df_anom = df_all[df_all['is_anomaly'] == 1].copy()
    print(f'  이상 샘플: {len(df_anom):,}행 '
          f'({len(df_anom)/len(df_all)*100:.2f}%)')

    if len(df_anom) == 0:
        print('  이상 샘플이 없습니다. Phase 5 결과 확인 필요.')
        return

    # ── 유형 분류 ──
    df_anom = classify_types(df_anom, df_all)
    log_step(f'유형 분류 완료: {len(df_anom):,}행')

    # ── 시각화 ──
    print('\n[시각화]')
    plot_type_pie(
        df_anom, os.path.join(FIG_DIR, 'phase06_type_pie.png')
    )
    plot_monthly_trend(
        df_anom, os.path.join(FIG_DIR, 'phase06_monthly_trend.png')
    )
    plot_branch_heatmap(
        df_anom, os.path.join(FIG_DIR, 'phase06_branch_heatmap.png')
    )

    # 대표 사례 (rep_index는 SHAP waterfall에서 재사용)
    rep_index = plot_representative_cases(df_anom, df_all, FIG_DIR)
    log_step('파이/월별/지사·종별/대표사례 시각화 완료')

    # ── SHAP ──
    shap_summary = run_shap(df_all, rep_index)
    if shap_summary:
        log_step(f'SHAP 완료: {len(shap_summary)}개 종별')

    # ── 검증용 샘플 (유형별 10건) ──
    review_rows = []
    for ptype, sub in df_anom.groupby('primary_type', observed=True):
        if ptype == '미분류':
            continue
        review_rows.append(
            sub.nlargest(10, 'severity')
        )
    if review_rows:
        review_df = pd.concat(review_rows, ignore_index=True)
        review_path = os.path.join(PHASE6_DIR, 'manual_review_samples.csv')
        review_cols = [
            FACILITY_COL, DATE_COL, '종별', '지사', '총사용량',
            'primary_type', 'secondary_types', 'severity',
            'if_score', 'ae_score', 'gmm_log_likelihood', 'context_zscore',
        ]
        review_cols = [c for c in review_cols if c in review_df.columns]
        review_df[review_cols].to_csv(
            review_path, index=False, encoding='utf-8-sig'
        )
        print(f'  수동 검토 샘플: {review_path} ({len(review_df)}건)')

    # ── 저장 ──
    print('\n[저장]')

    # 1) 유형 분류 결과 테이블 (분석플랜 6-5 산출물)
    out_cols = [
        FACILITY_COL, DATE_COL, '종별', '지사', '월', '요일',
        '난방시즌', '냉방시즌', '활성시즌',
        '총사용량',
        'if_score', 'if_flag', 'ae_score', 'ae_flag',
        'gmm_log_likelihood', 'context_zscore',
        'stat_zscore_flag', 'stat_iqr_flag',
        'anomaly_confidence', 'is_anomaly',
        'primary_type', 'secondary_types', 'severity',
    ]
    type_cols = [c for c in df_anom.columns if c.startswith('type_')]
    out_cols = [c for c in out_cols if c in df_anom.columns] + type_cols

    classified_path = f'{PROCESSED_DIR}/anomaly_classified.parquet'
    df_anom[out_cols].to_parquet(classified_path, index=False)
    size_mb = os.path.getsize(classified_path) / (1024 ** 2)
    print(f'  유형 분류 결과: {classified_path} '
          f'({len(df_anom):,}행, {size_mb:.1f} MB)')

    # 2) 유형별 분포 CSV
    type_counts = (
        df_anom.groupby(['종별', 'primary_type'], observed=True)
        .size().unstack(fill_value=0)
    )
    type_counts['전체'] = type_counts.sum(axis=1)
    counts_path = os.path.join(PHASE6_DIR, 'type_distribution.csv')
    type_counts.to_csv(counts_path, encoding='utf-8-sig')
    print(f'  유형 분포: {counts_path}')

    # 3) Phase 6 요약 JSON
    summary = {
        'anomaly_rows': int(len(df_anom)),
        'classified_columns': out_cols,
        'thresholds': {
            'SURGE_MA7_QTILE': SURGE_MA7_QTILE,
            'SURGE_Z_QTILE': SURGE_Z_QTILE,
            'NIGHT_SIGMA': NIGHT_SIGMA,
            'PATTERN_LL_QTILE': PATTERN_LL_QTILE,
            'PATTERN_AE_QTILE': PATTERN_AE_QTILE,
            'LONG_ZERO_DAYS': LONG_ZERO_DAYS,
            'LONG_MA7_QTILE': LONG_MA7_QTILE,
            'SEASON_REV_QTILE': SEASON_REV_QTILE,
            'WEEKEND_RATIO': WEEKEND_RATIO,
            'BASELOAD_QTILE': BASELOAD_QTILE,
        },
        'primary_type_counts': {
            k: int(v)
            for k, v in df_anom['primary_type'].value_counts().items()
        },
        'shap_top_features': shap_summary,
        'log': log,
    }
    summary_path = f'{PROCESSED_DIR}/phase6_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  요약: {summary_path}')

    # ── 최종 ──
    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Phase 6 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    for entry in log:
        print(f'  {entry}')
    print(f'\n  출력: {classified_path}')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    run()
