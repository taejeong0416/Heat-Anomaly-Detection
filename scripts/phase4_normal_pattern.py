"""
Phase 4. 정상 패턴 모델링
GMM(Gaussian Mixture Model)으로 종별×시즌별 정상 패턴 프로토타입을 정의하고,
log-likelihood를 이상 점수로, Context Z-score를 횡단면 비교 피처로 생성한다.

사용법:
  python scripts/phase4_normal_pattern.py
"""

import pandas as pd
import numpy as np
import json
import os
import gc
import time
import pickle
from sklearn.mixture import GaussianMixture

# ============================================================
# 0. 설정
# ============================================================
HOUR_COLS = [f'{i}시' for i in range(1, 25)]
HOUR_RATIO_COLS = [f'hour_ratio_{i}' for i in range(1, 25)]
FACILITY_COL = '설치'
DATE_COL = '날짜'

PROCESSED_DIR = 'data/processed'
FIG_DIR = 'outputs/figures'
GMM_DIR = 'outputs/phase4_gmm'

# 종별×시즌 8그룹 정의
GROUPS = [
    ('주택용', '난방시즌', lambda df: (df['종별'] == '주택용') & (df['난방시즌'] == True)),
    ('주택용', '비난방시즌', lambda df: (df['종별'] == '주택용') & (df['난방시즌'] == False)),
    ('업무용', '난방시즌', lambda df: (df['종별'] == '업무용') & (df['난방시즌'] == True)),
    ('업무용', '비난방시즌', lambda df: (df['종별'] == '업무용') & (df['난방시즌'] == False)),
    ('공공용', '난방시즌', lambda df: (df['종별'] == '공공용') & (df['난방시즌'] == True)),
    ('공공용', '비난방시즌', lambda df: (df['종별'] == '공공용') & (df['난방시즌'] == False)),
    ('냉수용', '냉방시즌', lambda df: (df['종별'] == '냉수용') & (df['냉방시즌'] == True)),
    ('냉수용', '비냉방시즌', lambda df: (df['종별'] == '냉수용') & (df['냉방시즌'] == False)),
]

# GMM 탐색 범위
K_RANGE_DEFAULT = range(2, 9)   # 난방 3종별
K_RANGE_COLD = range(2, 5)      # 냉수용 (소규모)
SAMPLE_SIZE = 150_000            # 학습용 샘플 크기
REG_COVAR = 1e-6                 # 공분산 정규화

log = []


def log_step(msg):
    log.append(msg)
    print(f'  → {msg}')


# ============================================================
# 1. GMM 클러스터링
# ============================================================

def fit_gmm_for_group(X, group_name, k_range, save_dir):
    """
    BIC 기반으로 최적 n_components를 선택하고 GMM을 학습한다.

    Parameters:
      X: hour_ratio 배열 (n_samples, 24), NaN 행 제외 상태
      group_name: 그룹 이름 (시각화/저장용)
      k_range: 탐색할 n_components 범위
      save_dir: 모델/차트 저장 디렉토리

    Returns:
      best_gmm: 학습된 GaussianMixture 모델
      best_k: 최적 n_components
      bic_scores: {k: bic} 딕셔너리
    """
    # 샘플 추출 (학습 속도)
    if len(X) > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), SAMPLE_SIZE, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    print(f'    학습 데이터: {len(X_sample):,}행 (전체 {len(X):,}행)')

    bic_scores = {}
    best_bic = np.inf
    best_gmm = None
    best_k = -1

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            reg_covar=REG_COVAR,
            max_iter=200,
            n_init=3,
            random_state=42,
        )
        gmm.fit(X_sample)
        bic = gmm.bic(X_sample)
        bic_scores[k] = bic

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

        print(f'      k={k}: BIC={bic:,.0f}')

    print(f'    → 최적 k={best_k} (BIC={best_bic:,.0f})')

    # 전체 데이터로 재학습 (샘플이 아닌 전체)
    if len(X) > SAMPLE_SIZE:
        print(f'    전체 데이터({len(X):,}행)로 재학습...')
        best_gmm = GaussianMixture(
            n_components=best_k,
            covariance_type='full',
            reg_covar=REG_COVAR,
            max_iter=200,
            n_init=3,
            random_state=42,
        )
        best_gmm.fit(X)

    # 모델 저장
    model_path = os.path.join(save_dir, f'gmm_model_{group_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_gmm, f)

    return best_gmm, best_k, bic_scores


def plot_bic_curve(bic_scores, group_name, save_dir):
    """BIC 곡선 시각화"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    ks = sorted(bic_scores.keys())
    bics = [bic_scores[k] for k in ks]
    best_k = ks[np.argmin(bics)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.7,
               label=f'Best k={best_k}')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('BIC')
    ax.set_title(f'{group_name} | GMM BIC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'phase04_bic_{group_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_cluster_patterns(gmm, group_name, save_dir):
    """클러스터별 24시간 평균 곡선 시각화"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    centers = gmm.means_  # (k, 24)
    weights = gmm.weights_
    k = len(weights)

    fig, ax = plt.subplots(figsize=(12, 6))
    hours = list(range(1, 25))

    for i in range(k):
        label = f'C{i} ({weights[i]*100:.1f}%)'
        ax.plot(hours, centers[i], 'o-', label=label, markersize=4)

    ax.set_xlabel('시간')
    ax.set_ylabel('시간대 비율 (hour_ratio)')
    ax.set_title(f'{group_name} | GMM 클러스터별 정상 패턴 (k={k})')
    ax.set_xticks(hours)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'phase04_clusters_{group_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()


# ============================================================
# 2. Context Z-score
# ============================================================

def compute_context_zscore(df):
    """
    종별×지사×월×요일 그룹별 median/MAD 기반 z-score 산출.

    Returns:
      df with 'context_zscore' column added
      group_stats: dict (JSON 저장용)
    """
    print('\n[Context Z-score 산출]')

    group_cols = ['종별', '지사', '월', '요일']
    usage_col = '총사용량'

    # 그룹별 median, MAD 계산
    grouped = df.groupby(group_cols, observed=True)[usage_col]
    median = grouped.transform('median')
    # MAD = median(|x - median(x)|)
    abs_dev = (df[usage_col] - median).abs()
    mad = df.assign(_abs_dev=abs_dev).groupby(
        group_cols, observed=True
    )['_abs_dev'].transform('median')
    df.drop(columns='_abs_dev', inplace=True, errors='ignore')

    # z-score = (x - median) / MAD, MAD=0이면 NaN
    mad_safe = mad.replace(0, np.nan)
    df['context_zscore'] = ((df[usage_col] - median) / mad_safe).astype(np.float32)

    # 그룹 통계 저장용 (테스트 데이터 적용)
    stats = df.groupby(group_cols, observed=True)[usage_col].agg(
        ['median', 'mad']
    ).reset_index()
    stats.columns = group_cols + ['median', 'mad']

    # JSON 직렬화 가능한 형태로 변환
    group_stats = {}
    for _, row in stats.iterrows():
        key = f"{row['종별']}_{row['지사']}_{int(row['월'])}_{int(row['요일'])}"
        group_stats[key] = {
            'median': round(float(row['median']), 6),
            'mad': round(float(row['mad']), 6),
        }

    n_valid = df['context_zscore'].notna().sum()
    n_total = len(df)
    print(f'  그룹 수: {len(group_stats):,}')
    print(f'  유효: {n_valid:,}/{n_total:,} ({n_valid/n_total*100:.1f}%)')

    return df, group_stats


# ============================================================
# 메인 파이프라인
# ============================================================

def run():
    t_start = time.time()

    print('=' * 60)
    print('Phase 4. 정상 패턴 모델링 (GMM + Context Z-score)')
    print('=' * 60)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(GMM_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    input_path = f'{PROCESSED_DIR}/features.parquet'

    import pyarrow.parquet as pq
    _table = pq.read_table(input_path)
    df = _table.to_pandas(self_destruct=True)
    del _table
    gc.collect()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f'  입력: {len(df):,}행, {len(df.columns)}개 컬럼')

    # ── 결과 컬럼 초기화 ──
    df['cluster_id'] = np.int8(-1)
    df['gmm_log_likelihood'] = np.float32(np.nan)

    # ── GMM 클러스터링 (종별×시즌 8그룹) ──
    print('\n[GMM 클러스터링]')

    gmm_summary = []

    for type_name, season_name, mask_fn in GROUPS:
        group_name = f'{type_name}_{season_name}'
        print(f'\n--- {group_name} ---')

        mask = mask_fn(df)
        n_group = mask.sum()
        print(f'  전체: {n_group:,}행')

        # hour_ratio 추출 + 제로 사용일 제외
        X_all = df.loc[mask, HOUR_RATIO_COLS].values.astype(np.float32)
        valid_mask = ~np.any(np.isnan(X_all), axis=1)
        n_valid = valid_mask.sum()
        n_zero = n_group - n_valid
        print(f'  유효: {n_valid:,}행 (제로 사용일 제외: {n_zero:,}행)')

        if n_valid < 100:
            print(f'  ⚠ 유효 데이터 부족, 건너뜀')
            log_step(f'{group_name}: SKIP (유효 {n_valid}행)')
            continue

        X_valid = X_all[valid_mask]
        valid_indices = df.index[mask][valid_mask]

        # k 범위 결정
        k_range = K_RANGE_COLD if '냉수용' in type_name else K_RANGE_DEFAULT

        # GMM 학습
        best_gmm, best_k, bic_scores = fit_gmm_for_group(
            X_valid, group_name, k_range, GMM_DIR
        )

        # 전체 유효 데이터에 predict
        cluster_ids = best_gmm.predict(X_valid)
        log_likelihoods = best_gmm.score_samples(X_valid).astype(np.float32)

        df.loc[valid_indices, 'cluster_id'] = cluster_ids.astype(np.int8)
        df.loc[valid_indices, 'gmm_log_likelihood'] = log_likelihoods

        # 시각화
        plot_bic_curve(bic_scores, group_name, FIG_DIR)
        plot_cluster_patterns(best_gmm, group_name, FIG_DIR)

        # 클러스터별 통계
        cluster_counts = pd.Series(cluster_ids).value_counts().sort_index()
        for ci, cnt in cluster_counts.items():
            pct = cnt / n_valid * 100
            print(f'    C{ci}: {cnt:,}행 ({pct:.1f}%)')

        gmm_summary.append({
            'group': group_name,
            'n_total': int(n_group),
            'n_valid': int(n_valid),
            'n_zero_excluded': int(n_zero),
            'best_k': int(best_k),
            'best_bic': float(bic_scores[best_k]),
            'log_likelihood_mean': float(np.mean(log_likelihoods)),
            'log_likelihood_q01': float(np.percentile(log_likelihoods, 1)),
            'log_likelihood_q05': float(np.percentile(log_likelihoods, 5)),
        })

        log_step(f'{group_name}: k={best_k}, 유효 {n_valid:,}행, '
                 f'제로 제외 {n_zero:,}행')

        del X_all, X_valid, cluster_ids, log_likelihoods
        gc.collect()

    # ── Context Z-score ──
    df, group_stats = compute_context_zscore(df)
    log_step(f'Context Z-score 완료: {len(group_stats):,}개 그룹')

    # ── 저장 ──
    print('\n[저장]')

    # 피처 데이터
    output_path = f'{PROCESSED_DIR}/features_phase4.parquet'
    df.to_parquet(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 ** 2)
    print(f'  피처 데이터: {output_path} ({file_size:.1f} MB)')
    print(f'  컬럼: {len(df.columns)}개')

    # GMM 요약
    gmm_summary_df = pd.DataFrame(gmm_summary)
    gmm_summary_df.to_csv(
        os.path.join(GMM_DIR, 'gmm_summary.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f'  GMM 요약: {GMM_DIR}/gmm_summary.csv')

    # 그룹 통계 JSON
    stats_path = f'{PROCESSED_DIR}/group_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(group_stats, f, ensure_ascii=False, indent=2)
    print(f'  그룹 통계: {stats_path}')

    # Phase 4 요약 JSON
    summary = {
        'input_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'new_columns': ['cluster_id', 'gmm_log_likelihood', 'context_zscore'],
        'gmm_groups': gmm_summary,
        'context_zscore_groups': len(group_stats),
        'log': log,
    }
    summary_path = f'{PROCESSED_DIR}/phase4_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  요약: {summary_path}')

    # ── 최종 요약 ──
    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Phase 4 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    for entry in log:
        print(f'  {entry}')
    print(f'\n  출력: {output_path} ({file_size:.1f} MB)')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    run()
