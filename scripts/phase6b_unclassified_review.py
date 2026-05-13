"""
Phase 6b. 미분류 이상 샘플 수동 검토
주택용 미분류 (primary_type=='미분류') 케이스에서 50건을 stratified 추출하여
24시간 사용량 곡선 + 정상 평균 오버레이 + 점수표를 시각화·요약한다.

목적:
  - 미분류 케이스의 정체 파악 → 신규 유형 도출 or "약한 이상 신호" 결론
  - 데이터 기반 의사결정 근거 확보 (이상유형정의.md 갱신용)

사용법:
  python scripts/phase6b_unclassified_review.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gc

HOUR_COLS = [f'{i}시' for i in range(1, 25)]
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED = 'data/processed'
FIG_DIR = 'outputs/figures'
PHASE6_DIR = 'outputs/phase6'

REVIEW_TARGET_TYPE = '주택용'     # 미분류가 가장 많은 종별
SAMPLE_PER_CONFIDENCE = {
    'high': 15,
    'low': 15,
    'stat_only': 20,
}
RANDOM_STATE = 42


def setup_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False


def plot_grid(sample_df, normal_curve, save_path):
    """샘플 50건의 24시간 곡선을 5×10 grid로 시각화."""
    n = len(sample_df)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    hours = list(range(1, 25))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[i]
        ax.plot(hours, row[HOUR_COLS].values, '-', color='#E53935',
                linewidth=1.2, label='이상')
        ax.plot(hours, normal_curve.values, '--', color='#1E88E5',
                linewidth=0.8, alpha=0.7, label='정상 평균')
        title = (
            f"#{i + 1} {row['anomaly_confidence']}\n"
            f"IF={row['if_score']:.2f} "
            f"AE={row['ae_score']:.4f}\n"
            f"MP={row['mp_score']:.2f} "
            f"z={row['context_zscore']:.1f}"
        )
        ax.set_title(title, fontsize=7)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'{REVIEW_TARGET_TYPE} 미분류 샘플 {n}건 (24시간 곡선)',
        fontsize=12, y=1.001,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def summarize_patterns(sample_df, df_all):
    """샘플의 통계적 특성 요약."""
    feats = [
        '총사용량', 'cv', 'hourly_std', 'baseload',
        'night_ratio', 'day_ratio', 'peak_hour',
        'ma7_ratio', 'context_zscore',
        'if_score', 'ae_score', 'mp_score',
        'gmm_log_likelihood',
    ]
    feats = [f for f in feats if f in sample_df.columns]

    # 종별 활성시즌 정상 분포와 비교
    normal_mask = (
        (df_all['종별'] == REVIEW_TARGET_TYPE)
        & (df_all['is_anomaly'] == 0)
    )
    normal_stats = (
        df_all.loc[normal_mask, feats]
        .agg(['mean', 'std', 'median'])
        .T
    )

    sample_stats = (
        sample_df[feats]
        .agg(['mean', 'std', 'median'])
        .T
        .rename(columns={
            'mean': 'sample_mean', 'std': 'sample_std',
            'median': 'sample_median',
        })
    )

    cmp = pd.concat([
        normal_stats.rename(columns={
            'mean': 'normal_mean', 'std': 'normal_std',
            'median': 'normal_median',
        }),
        sample_stats,
    ], axis=1)
    cmp['z_vs_normal'] = (
        (cmp['sample_mean'] - cmp['normal_mean']) / cmp['normal_std']
    )
    return cmp


def main():
    print('=' * 60)
    print(f'Phase 6b. {REVIEW_TARGET_TYPE} 미분류 샘플 수동 검토')
    print('=' * 60)
    setup_font()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PHASE6_DIR, exist_ok=True)

    print('\n[데이터 로드]')
    # 분류 결과
    clf = pd.read_parquet(f'{PROCESSED}/anomaly_classified.parquet')
    clf[DATE_COL] = pd.to_datetime(clf[DATE_COL])
    print(f'  분류 결과: {len(clf):,}행')

    # 전체 (정상 평균 곡선용 + 분포 비교용)
    df_all = pd.read_parquet(
        f'{PROCESSED}/features_phase5.parquet',
        columns=(
            HOUR_COLS + [
                FACILITY_COL, DATE_COL, '종별', '활성시즌'
                if '활성시즌' in pd.read_parquet(
                    f'{PROCESSED}/features_phase5.parquet',
                    columns=['난방시즌'],
                ).columns else '난방시즌',
                '난방시즌', '냉방시즌',
                'is_anomaly', '총사용량',
                'cv', 'hourly_std', 'baseload',
                'night_ratio', 'day_ratio', 'peak_hour',
                'ma7_ratio', 'context_zscore',
                'if_score', 'ae_score', 'mp_score',
                'gmm_log_likelihood',
            ]
        ) if False else None,   # 위 expression은 정확하지 않으므로 fallback
    )
    # 위 hack 제거 후 단순 로드
    df_all = pd.read_parquet(f'{PROCESSED}/features_phase5.parquet')
    df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL])
    # 활성시즌 부여
    is_cold = df_all['종별'] == '냉수용'
    df_all['활성시즌'] = np.where(
        is_cold, df_all['냉방시즌'].astype(bool),
        df_all['난방시즌'].astype(bool),
    )
    print(f'  전체: {len(df_all):,}행')

    # ── 미분류 주택용 추출 ──
    target = clf[
        (clf['종별'] == REVIEW_TARGET_TYPE)
        & (clf['primary_type'] == '미분류')
    ]
    print(f'  {REVIEW_TARGET_TYPE} 미분류: {len(target):,}행')

    # ── stratified 샘플 ──
    samples = []
    for conf_name, n_pick in SAMPLE_PER_CONFIDENCE.items():
        sub = target[target['anomaly_confidence'] == conf_name]
        if len(sub) == 0:
            print(f'  {conf_name}: 0행 (skip)')
            continue
        n = min(n_pick, len(sub))
        s = sub.sample(n, random_state=RANDOM_STATE)
        samples.append(s)
        print(f'  {conf_name}: {n}건 추출 (모집단 {len(sub):,})')
    sample_df = pd.concat(samples, ignore_index=True)
    print(f'  총 표본: {len(sample_df)}건')

    # ── 샘플의 24시간 데이터 가져오기 (features_phase5에서 lookup) ──
    keys = list(zip(sample_df[FACILITY_COL], sample_df[DATE_COL]))
    df_all_idx = df_all.set_index([FACILITY_COL, DATE_COL])
    # 일부 키가 안 맞을 수 있으니 inner join으로
    sample_full = (
        df_all_idx.loc[df_all_idx.index.isin(keys)]
        .reset_index()
    )
    # features_phase5에 anomaly_confidence 컬럼이 있으면 그대로 사용
    if 'anomaly_confidence' not in sample_full.columns:
        sample_df_merged = sample_df[[
            FACILITY_COL, DATE_COL, 'anomaly_confidence'
        ]].merge(sample_full, on=[FACILITY_COL, DATE_COL], how='inner')
    else:
        sample_df_merged = sample_full.merge(
            sample_df[[FACILITY_COL, DATE_COL]],
            on=[FACILITY_COL, DATE_COL], how='inner',
        )
    print(f'  hour 데이터 매칭: {len(sample_df_merged)}건')

    # ── 정상 평균 곡선 (주택용 활성시즌=난방시즌 정상) ──
    normal_curve_h = (
        df_all[
            (df_all['종별'] == REVIEW_TARGET_TYPE)
            & (df_all['활성시즌'] == True)
            & (df_all['is_anomaly'] == 0)
        ][HOUR_COLS].mean()
    )
    normal_curve_nh = (
        df_all[
            (df_all['종별'] == REVIEW_TARGET_TYPE)
            & (df_all['활성시즌'] == False)
            & (df_all['is_anomaly'] == 0)
        ][HOUR_COLS].mean()
    )

    # ── 활성/비활성 별도 시각화 ──
    print('\n[시각화]')
    act = sample_df_merged[sample_df_merged['활성시즌'] == True]
    inact = sample_df_merged[sample_df_merged['활성시즌'] == False]
    print(f'  활성시즌: {len(act)}, 비활성시즌: {len(inact)}')

    if len(act):
        plot_grid(act, normal_curve_h,
                  f'{FIG_DIR}/phase06b_unclassified_active.png')
        print(f'  → phase06b_unclassified_active.png ({len(act)}건)')
    if len(inact):
        plot_grid(inact, normal_curve_nh,
                  f'{FIG_DIR}/phase06b_unclassified_inactive.png')
        print(f'  → phase06b_unclassified_inactive.png ({len(inact)}건)')

    # ── 분포 비교표 ──
    print('\n[통계 요약 (샘플 vs 정상 분포)]')
    cmp = summarize_patterns(sample_df_merged, df_all)
    print(cmp.round(3).to_string())
    cmp_path = f'{PHASE6_DIR}/unclassified_sample_stats.csv'
    cmp.to_csv(cmp_path, encoding='utf-8-sig')
    print(f'\n  → {cmp_path}')

    # ── 샘플 raw 데이터 CSV ──
    out_cols = [
        FACILITY_COL, DATE_COL, '종별', '지사', '월', '요일',
        '활성시즌', 'anomaly_confidence',
        '총사용량', 'cv', 'hourly_std', 'baseload',
        'night_ratio', 'day_ratio', 'peak_hour',
        'ma7_ratio', 'context_zscore',
        'if_score', 'ae_score', 'mp_score',
        'gmm_log_likelihood',
    ]
    out_cols = [c for c in out_cols if c in sample_df_merged.columns]
    sample_path = f'{PHASE6_DIR}/unclassified_samples.csv'
    sample_df_merged[out_cols].to_csv(
        sample_path, index=False, encoding='utf-8-sig',
    )
    print(f'  → {sample_path} ({len(sample_df_merged)}건)')

    print('\n[해석 가이드]')
    print('  - 시각화 PNG를 열어 24시간 곡선 패턴을 검토')
    print('  - 정상 평균(파란 점선) 대비 형태 차이가 크면 새 유형 후보')
    print('  - 거의 비슷하면 "약한 이상 신호 — 추가 검토 필요"로 결론')
    print('  - 통계 요약(unclassified_sample_stats.csv)의 z_vs_normal에서')
    print('    크게 벗어난 컬럼이 보이면 그 신호 위주로 유형 정의 검토')


if __name__ == '__main__':
    main()
