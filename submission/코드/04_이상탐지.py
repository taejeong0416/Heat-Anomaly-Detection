"""
Phase 5. 이상 탐지
Phase 4의 정상 패턴 모델 출력(gmm_log_likelihood, context_zscore, cluster_id)에
Isolation Forest + Autoencoder + Matrix Profile + 통계 기반 보조 탐지를 결합하여
설비-일 단위 anomaly score와 이상 여부 라벨을 산출한다.

학술/도메인 근거:
  Liu et al. (2008)  — Isolation Forest
  Hinton & Salakhutdinov (2006), Sakurada & Yairi (2014) — Autoencoder 이상 탐지
  Yeh et al. (2016) — Matrix Profile (시계열 자기 참조 이상 탐지)
  PNNL-24331 (2015) — z-score / IQR 기반 보조 필터
  도메인조사.md §5 — 이상 7유형 (Phase 5-1)

설계 결정:
  - 스케일러: heavy-tailed 분포(총사용량, log-likelihood 등)에 강건한
    RobustScaler(median/IQR) 사용. StandardScaler는 극단값 영향으로
    일반 행이 0 근처로 압축되어 부적절.
  - 임계 선정: 라벨이 없으므로 Phase 4의 GMM log-likelihood 하위 q%
    를 약 라벨(weak label)로 사용하여 F1 최대화 하는 후보를 선택한다.
    GMM은 본 프로젝트의 "정상 패턴 프로토타입"이므로 도메인적으로
    합리적인 reference. 선정 결과는 phase5_summary.json에 기록.

사용법:
  python scripts/phase5_anomaly_detection.py
"""

import pandas as pd
import numpy as np
import json
import os
import gc
import time
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# Matrix Profile
try:
    import stumpy
except ImportError:
    raise ImportError(
        'Matrix Profile 계산에 stumpy 패키지 필요: pip install stumpy'
    )

# Autoencoder는 PyTorch 사용 (sklearn에는 적합한 AE 구현 없음)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

# 종별 (4그룹) — Isolation Forest / AE 종별 분리 학습
TYPES = ['주택용', '업무용', '공공용', '냉수용']

# ── 그리드 탐색 후보 ──
# IF contamination: 분석플랜 5-2 명시 범위 0.01~0.05
IF_CONTAMINATION_GRID = [0.01, 0.02, 0.03, 0.05]
# AE 복원 오차 상위 N%
AE_TOP_PCT_GRID = [0.01, 0.02, 0.03, 0.05]
# 통계 |z| 상위 N%
STAT_TOP_PCT_GRID = [0.005, 0.01, 0.02, 0.03]

# Weak label: GMM log-likelihood 하위 q%를 "참 이상"으로 간주
# 분석플랜 5-2 범위 중간값 사용 (선정 기준이지 학습용 임계가 아님)
WEAK_LABEL_QUANTILE = 0.03

IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256
IF_SAMPLE_SIZE = 200_000   # IF 학습용 샘플 상한 (종별별)

# Autoencoder 구조 (분석플랜 5-3)
AE_INPUT_DIM = 24
AE_HIDDEN = [16, 8, 16]
AE_EPOCHS = 30
AE_BATCH = 1024
AE_LR = 1e-3
AE_SAMPLE_SIZE = 300_000   # AE 학습용 샘플 상한 (종별별)
AE_VAL_RATIO = 0.1

# IF 입력 피처 (분석플랜 5-2)
# gmm_log_likelihood 제거: Phase 4 모델 출력을 IF 입력에 넣으면 정보 누수(leakage)
# context_zscore는 그룹 통계 기반이므로 보조 피처로 유지
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

# Matrix Profile 설정 (stumpy 기반 시계열 자기 참조 이상 탐지)
# 설비별 일별 총사용량 시계열 → window=7(주간 패턴)로 discord 탐지
# GMM/AE는 "다른 설비 대비" 비교, MP는 "자기 과거 대비" 비교 → 관점 차별화
MP_WINDOW = 7          # 7일 = 주간 패턴 (요일 주기 포착)
MP_MIN_DAYS = 14       # 최소 2주 이상 데이터 필요
MP_TOP_PCT_GRID = [0.01, 0.02, 0.03, 0.05]

RANDOM_STATE = 42
log = []


def log_step(msg):
    log.append(msg)
    print(f'  → {msg}')


# ============================================================
# 1. 그리드 탐색 유틸 (weak label F1)
# ============================================================

def weak_label_from_gmm(loglik, quantile=WEAK_LABEL_QUANTILE):
    """
    GMM log-likelihood 하위 q%를 약 라벨(이상=1)로 정의.
    NaN(GMM 미적용 행)은 0으로 둔다.

    주의(순환 논리 한계):
      이 약 라벨은 IF/AE 모델 '학습'이 아닌 '임계 선정'에만 사용.
      IF/AE는 비지도 학습이며, 약 라벨은 여러 후보 임계 중
      GMM 정상 패턴과 가장 일치하는 것을 고르는 역할.
      라벨 없는 환경에서의 실용적 접근이나, 최종 검증은
      Phase 6 수동 검토(manual_review_samples)로 보완.
    """
    valid = np.isfinite(loglik)
    y = np.zeros(len(loglik), dtype=np.int8)
    if valid.sum() == 0:
        return y
    thr = np.quantile(loglik[valid], quantile)
    y[valid & (loglik <= thr)] = 1
    return y


def grid_select_by_f1(score, weak_label, candidates):
    """
    score(클수록 이상)에 대해 후보 top_pct를 적용했을 때
    weak_label과의 F1을 측정하여 최대값의 후보를 반환.

    Returns:
      best_q, results (list of dicts), threshold(at best_q)
    """
    valid = np.isfinite(score) & (weak_label >= 0)
    s = score[valid]
    y = weak_label[valid]
    results = []
    best = None
    for q in candidates:
        thr = np.quantile(s, 1 - q)
        pred = (s >= thr).astype(np.int8)
        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        results.append({
            'top_pct': float(q),
            'threshold': float(thr),
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'n_anomaly': int(pred.sum()),
        })
        if best is None or f1 > best['f1']:
            best = {'top_pct': float(q), 'threshold': float(thr),
                    'f1': float(f1)}
    return best, results


# ============================================================
# 2. Isolation Forest (종별별 학습 + contamination 그리드)
# ============================================================

def fit_isolation_forest(X, contamination, save_path):
    """
    Isolation Forest 학습 + 모델 저장.

    Parameters:
      X: (n_samples, n_features) RobustScaler 적용 입력
      contamination: 이상 비율
      save_path: 모델 저장 경로 (.pkl)

    Returns:
      model: 학습된 IsolationForest
    """
    n = len(X)
    if n > IF_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, IF_SAMPLE_SIZE, replace=False)
        X_fit = X[idx]
        print(f'    학습 샘플: {len(X_fit):,}행 (전체 {n:,}행)')
    else:
        X_fit = X
        print(f'    학습 샘플: {n:,}행')

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    return model


def score_isolation_forest(model, X):
    """
    Isolation Forest score 산출.
      score = -decision_function (값이 클수록 이상)
    """
    raw = model.decision_function(X)        # 클수록 정상
    score = (-raw).astype(np.float32)        # 클수록 이상
    return score


def grid_search_if(X, weak_label, save_dir, type_name):
    """
    IF는 contamination이 학습 시 트리 분할이 아닌 임계 결정에만 영향.
    하나의 모델을 학습하고 score 분포에서 후보 contamination을 임계로
    적용해 F1을 비교 → 최적 선정. 학습 비용 절감.
    """
    print(f'  [IF 그리드 탐색]')
    # 기본 학습은 0.02로 (임계는 어차피 후처리)
    model_path = os.path.join(save_dir, f'if_{type_name}.pkl')
    model = fit_isolation_forest(X, 0.02, model_path)
    score = score_isolation_forest(model, X)
    best, results = grid_select_by_f1(score, weak_label, IF_CONTAMINATION_GRID)
    print(f'    contamination 후보별 F1:')
    for r in results:
        mark = ' ◀ best' if r['top_pct'] == best['top_pct'] else ''
        print(f"      c={r['top_pct']:.3f}: "
              f"F1={r['f1']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f}{mark}")
    return model, score, best, results


# ============================================================
# 2. Autoencoder (종별별 학습)
# ============================================================

class Autoencoder(nn.Module):
    """24차원 입력 → Dense(16) → Dense(8) → Dense(16) → 24차원 출력."""

    def __init__(self, input_dim=AE_INPUT_DIM, hidden=AE_HIDDEN):
        super().__init__()
        h1, h2, h3 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit_autoencoder(X, save_path):
    """
    Autoencoder 학습 + 모델 저장.
      입력: hour_ratio 24차원 (행 합 ≈ 1)
      손실: MSE
    """
    n = len(X)
    if n > AE_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, AE_SAMPLE_SIZE, replace=False)
        X_fit = X[idx]
        print(f'    학습 샘플: {len(X_fit):,}행 (전체 {n:,}행)')
    else:
        X_fit = X
        print(f'    학습 샘플: {n:,}행')

    # train/val 분할
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(X_fit))
    n_val = int(len(X_fit) * AE_VAL_RATIO)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr = torch.from_numpy(X_fit[tr_idx]).float()
    X_val = torch.from_numpy(X_fit[val_idx]).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    device: {device}')

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr), batch_size=AE_BATCH,
        shuffle=True, num_workers=0,
    )

    best_val = np.inf
    best_state = None
    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        tr_loss = total_loss / len(X_tr)

        model.eval()
        with torch.no_grad():
            xv = X_val.to(device)
            val_loss = loss_fn(model(xv), xv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'    epoch {epoch+1:2d}/{AE_EPOCHS}: '
                  f'train={tr_loss:.6f}, val={val_loss:.6f}')

    # best state 복원 후 저장
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(),
                'input_dim': AE_INPUT_DIM,
                'hidden': AE_HIDDEN},
               save_path)
    print(f'    best val loss: {best_val:.6f}')

    return model, device


def score_autoencoder(model, device, X, batch=4096):
    """
    AE 복원 오차(MSE per row) 산출.
    """
    model.eval()
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            xb = torch.from_numpy(X[s:e]).float().to(device)
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=1)
            out[s:e] = err.detach().cpu().numpy()
    return out


# ============================================================
# 3. 통계 기반 보조 탐지
# ============================================================

def stat_anomaly_flags(df, weak_label):
    """
    Phase 5-4. 통계 보조 탐지 (그리드 탐색으로 임계 선정).
      - |context_zscore| 상위 q% → stat_zscore_flag (q는 STAT_TOP_PCT_GRID에서 F1 최대 선정)
      - 종별·월 IQR (Tukey 상한, 통계학 표준 1.5*IQR) → stat_iqr_flag
        IQR 룰은 분포 가정 없이 종별·월 컨텍스트에 적응적이므로 그대로 사용.
    """
    print('\n[통계 기반 보조 탐지]')

    # ── z-score 그리드 탐색 ──
    z = df['context_zscore'].abs().values.astype(np.float32)
    best_z, results_z = grid_select_by_f1(z, weak_label, STAT_TOP_PCT_GRID)
    print('  |context_zscore| 후보별 F1:')
    for r in results_z:
        mark = ' ◀ best' if r['top_pct'] == best_z['top_pct'] else ''
        print(f"    q={r['top_pct']:.4f}: F1={r['f1']:.3f} "
              f"P={r['precision']:.3f} R={r['recall']:.3f}{mark}")
    thr_z = best_z['threshold']
    stat_zscore_flag = np.zeros(len(df), dtype=np.int8)
    valid_z = np.isfinite(z)
    stat_zscore_flag[valid_z & (z >= thr_z)] = 1
    log_step(f"z-score 선정: 상위 {best_z['top_pct']*100:.2f}% "
             f"(|z|≥{thr_z:.2f}, F1={best_z['f1']:.3f}) → "
             f"이상 {int(stat_zscore_flag.sum()):,}행")

    # ── 종별·지사·월 IQR (Tukey 1.5×IQR 표준 룰) ──
    grp = df.groupby(['종별', '지사', '월'], observed=True)['총사용량']
    q1 = grp.transform(lambda x: x.quantile(0.25))
    q3 = grp.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    stat_iqr_flag = (df['총사용량'] > upper).astype(np.int8).values
    log_step(f'IQR(Tukey 1.5×IQR) 상위 이상 {int(stat_iqr_flag.sum()):,}행')

    df['stat_zscore_flag'] = stat_zscore_flag
    df['stat_iqr_flag'] = stat_iqr_flag
    return df, {
        'zscore_grid': results_z,
        'zscore_best': best_z,
    }


# ============================================================
# 4. Matrix Profile (설비별 시계열 자기 참조 이상 탐지)
# ============================================================

def compute_matrix_profile(df):
    """
    설비별 일별 총사용량 시계열에 Matrix Profile (stumpy) 적용.
    window=7 (주간 패턴): "이 설비의 이번 주 사용 궤적이 과거와 얼마나 다른가"를 측정.

    기존 방법과의 차별점:
      GMM         → 같은 종별의 다른 설비들 대비 (횡단면, 인구 참조)
      Context Z   → 같은 조건 동료 건물 대비 (횡단면, 동료 참조)
      AE          → 전체 정상 패턴 대비 복원 오차 (패턴 형태)
      **MP**      → **자기 자신의 과거** 대비 (시계열, 자기 참조)

    Returns: np.float32 array (length=len(df)), 값이 클수록 이질적 구간.
    """
    print('\n[Matrix Profile 계산]')
    mp_scores = np.full(len(df), np.nan, dtype=np.float32)

    t0 = time.time()
    n_fac = df[FACILITY_COL].nunique()
    processed = 0
    skipped = 0

    for fac, grp in df.groupby(FACILITY_COL):
        grp_sorted = grp.sort_values(DATE_COL)
        ts = grp_sorted['총사용량'].values.astype(np.float64)
        n_days = len(ts)

        if n_days < MP_MIN_DAYS:
            skipped += 1
            continue

        # NaN 처리: 중위수로 채움 (MP 계산용, 결과에선 원래 NaN 위치 복원)
        nan_mask = ~np.isfinite(ts)
        if nan_mask.all():
            skipped += 1
            continue
        ts_clean = ts.copy()
        if nan_mask.any():
            ts_clean[nan_mask] = np.nanmedian(ts)

        # 상수 시계열 (std≈0) → stumpy 정규화 실패 방지
        if np.std(ts_clean) < 1e-10:
            skipped += 1
            continue

        # stumpy Matrix Profile (window=7, 주간 패턴)
        result = stumpy.stump(ts_clean, m=MP_WINDOW)
        mp_vals = result[:, 0].astype(np.float32)

        # MP 길이 = n_days - MP_WINDOW + 1
        # 각 MP[i]를 i번째 날에 할당 (7일 구간의 시작일)
        day_scores = np.full(n_days, np.nan, dtype=np.float32)
        day_scores[:len(mp_vals)] = mp_vals
        # 원래 NaN이었던 날은 다시 NaN
        day_scores[nan_mask] = np.nan

        mp_scores[grp_sorted.index] = day_scores
        processed += 1

        if processed % 2000 == 0:
            elapsed = time.time() - t0
            print(f'    {processed}/{n_fac} 설비 처리 ({elapsed:.1f}s)')

    elapsed = time.time() - t0
    valid_count = int(np.isfinite(mp_scores).sum())
    print(f'  완료: {processed}설비 처리, {skipped}설비 스킵 ({elapsed:.1f}s)')
    print(f'  유효 mp_score: {valid_count:,}행 ({valid_count/len(df)*100:.1f}%)')
    log_step(f'Matrix Profile 완료: {processed}설비, 유효 {valid_count:,}행 ({elapsed:.1f}s)')

    return mp_scores


# ============================================================
# 5. 시각화
# ============================================================

def plot_score_dist(scores, title, save_path, thr=None):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(finite, bins=80, color='#FF7043', alpha=0.85)
    if thr is not None:
        ax.axvline(thr, color='red', linestyle='--',
                   label=f'threshold={thr:.4f}')
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('score')
    ax.set_ylabel('count')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ensemble_overlap(df, save_path):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    vote = (df['if_flag'].astype(int)
            + df['ae_flag'].astype(int)
            + df['mp_flag'].astype(int))
    cnts = {
        '0표 (정상)': int((vote == 0).sum()),
        '1표 (저신뢰)': int((vote == 1).sum()),
        '2표 (고신뢰)': int((vote == 2).sum()),
        '3표 (최고신뢰)': int((vote == 3).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(cnts.keys())
    vals = [cnts[k] for k in keys]
    colors = ['#BDBDBD', '#42A5F5', '#EF5350', '#B71C1C']
    ax.bar(keys, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    ax.set_title('IF / AE / MP 앙상블 투표 분포')
    ax.set_ylabel('rows')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 5. 메인 파이프라인
# ============================================================

def run():
    t_start = time.time()

    print('=' * 60)
    print('Phase 5. 이상 탐지 (IF + AE + Matrix Profile + 통계)')
    print('=' * 60)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

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

    # ── 입력 데이터 검증 ──
    required_cols = IF_FEATURES + ['gmm_log_likelihood', 'cluster_id']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f'Phase 4 산출물 컬럼 누락: {missing}. '
            f'Phase 4 (scripts/phase4_normal_pattern.py) 실행 후 재시도.'
        )

    # 결측 보호: IF 입력에서 NaN 행 인덱싱 위해 마스크 보관
    missing_required = df[IF_FEATURES].isna().any(axis=1)
    print(f'  IF 입력 NaN 행: {int(missing_required.sum()):,}'
          f' ({missing_required.mean()*100:.2f}%)')

    # ── 결과 컬럼 초기화 ──
    df['if_score'] = np.float32(np.nan)
    df['if_flag'] = np.int8(0)
    df['ae_score'] = np.float32(np.nan)
    df['ae_flag'] = np.int8(0)
    df['mp_score'] = np.float32(np.nan)
    df['mp_flag'] = np.int8(0)

    # ── 종별별 Isolation Forest + Autoencoder ──
    print('\n[종별별 IF + AE 학습 + 그리드 탐색]')
    summary_rows = []
    grid_records = {}   # 종별별 그리드 결과 (summary JSON에 기록)

    for type_name in TYPES:
        print(f'\n--- {type_name} ---')
        mask = (df['종별'] == type_name) & (~missing_required)
        n_group = int(mask.sum())
        print(f'  유효 행: {n_group:,}')

        if n_group < 1000:
            log_step(f'{type_name}: SKIP (유효 {n_group}행)')
            continue

        idx = df.index[mask]
        record = {'n_valid': n_group}

        # ── weak label: 해당 종별의 GMM log-likelihood 하위 q% ──
        loglik_grp = df.loc[mask, 'gmm_log_likelihood'].values
        weak_y = weak_label_from_gmm(loglik_grp, WEAK_LABEL_QUANTILE)
        n_pos = int(weak_y.sum())
        print(f'  weak label(GMM 하위 {WEAK_LABEL_QUANTILE*100:.0f}%): '
              f'{n_pos:,}행 ({n_pos/n_group*100:.2f}%)')

        # ----- (a) Isolation Forest -----
        X_if_raw = df.loc[mask, IF_FEATURES].values.astype(np.float32)
        # RobustScaler: heavy-tailed 피처에 적합 (median/IQR)
        scaler = RobustScaler()
        X_if = scaler.fit_transform(X_if_raw).astype(np.float32)
        scaler_path = os.path.join(MODEL_DIR, f'if_scaler_{type_name}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        if_model, if_score, if_best, if_grid = grid_search_if(
            X_if, weak_y, MODEL_DIR, type_name
        )
        thr_if = if_best['threshold']
        if_flag_local = (if_score >= thr_if).astype(np.int8)
        df.loc[idx, 'if_score'] = if_score
        df.loc[idx, 'if_flag'] = if_flag_local
        n_if_anom = int(if_flag_local.sum())
        print(f"    ▶ 선정 contamination={if_best['top_pct']:.3f} "
              f"(F1={if_best['f1']:.3f}), "
              f"이상 {n_if_anom:,}행 ({n_if_anom/n_group*100:.2f}%)")
        record['if_grid'] = if_grid
        record['if_best'] = if_best

        plot_score_dist(
            if_score, f'{type_name} | IF anomaly score',
            os.path.join(FIG_DIR, f'phase05_if_score_{type_name}.png'),
            thr=thr_if,
        )

        # ----- (b) Autoencoder (hour_ratio 24차원) -----
        X_ae = df.loc[mask, HOUR_RATIO_COLS].values.astype(np.float32)
        nz = X_ae.sum(axis=1) > 0
        if nz.sum() < 1000:
            log_step(f'{type_name}: AE SKIP (유효 비제로 {int(nz.sum())}행)')
            n_ae_anom = 0
            ae_best = None
        else:
            ae_model_path = os.path.join(MODEL_DIR, f'ae_{type_name}.pt')
            print('  [Autoencoder]')
            ae_model, device = fit_autoencoder(X_ae[nz], ae_model_path)
            ae_err = score_autoencoder(ae_model, device, X_ae[nz])

            full_err = np.full(len(X_ae), np.nan, dtype=np.float32)
            full_err[nz] = ae_err
            df.loc[idx, 'ae_score'] = full_err

            # AE top_pct 그리드 탐색
            ae_weak = weak_y[nz]
            ae_best, ae_grid = grid_select_by_f1(
                ae_err, ae_weak, AE_TOP_PCT_GRID
            )
            print('  [AE 그리드 탐색]')
            for r in ae_grid:
                mark = ' ◀ best' if r['top_pct'] == ae_best['top_pct'] else ''
                print(f"      q={r['top_pct']:.3f}: "
                      f"F1={r['f1']:.3f} P={r['precision']:.3f} "
                      f"R={r['recall']:.3f}{mark}")
            thr_ae = ae_best['threshold']
            ae_flag_local = np.zeros(len(X_ae), dtype=np.int8)
            ae_flag_local[nz] = (ae_err >= thr_ae).astype(np.int8)
            df.loc[idx, 'ae_flag'] = ae_flag_local
            n_ae_anom = int(ae_flag_local.sum())
            print(f"    ▶ 선정 top_pct={ae_best['top_pct']:.3f} "
                  f"(F1={ae_best['f1']:.3f}, thr={thr_ae:.6f}), "
                  f"이상 {n_ae_anom:,}행 ({n_ae_anom/n_group*100:.2f}%)")
            record['ae_grid'] = ae_grid
            record['ae_best'] = ae_best

            plot_score_dist(
                ae_err, f'{type_name} | AE reconstruction error',
                os.path.join(FIG_DIR, f'phase05_ae_score_{type_name}.png'),
                thr=thr_ae,
            )

            del ae_model, ae_err, full_err
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary_rows.append({
            'type': type_name,
            'n_valid': n_group,
            'if_contamination': if_best['top_pct'],
            'if_f1': if_best['f1'],
            'if_anomalies': n_if_anom,
            'if_anomaly_pct': n_if_anom / n_group * 100,
            'ae_top_pct': ae_best['top_pct'] if ae_best else None,
            'ae_f1': ae_best['f1'] if ae_best else None,
            'ae_anomalies': n_ae_anom,
            'ae_anomaly_pct': n_ae_anom / n_group * 100 if n_group else 0,
        })
        grid_records[type_name] = record

        log_step(f"{type_name}: 유효 {n_group:,}행, "
                 f"IF c={if_best['top_pct']:.3f}(F1={if_best['f1']:.3f}) "
                 f"이상 {n_if_anom:,}행")

        del X_if_raw, X_if, if_score, if_flag_local
        gc.collect()

    # ── Matrix Profile (설비별 시계열 자기 참조 이상 탐지) ──
    df['mp_score'] = compute_matrix_profile(df)

    # 종별별 MP 그리드 탐색 (weak label 기반 F1 최대화)
    print('\n[종별별 MP 그리드 탐색]')
    full_weak = weak_label_from_gmm(
        df['gmm_log_likelihood'].values, WEAK_LABEL_QUANTILE
    )
    for type_name in TYPES:
        mask = df['종별'] == type_name
        mp_vals = df.loc[mask, 'mp_score'].values.astype(np.float32)
        valid_mp = np.isfinite(mp_vals)

        if valid_mp.sum() < 1000:
            log_step(f'{type_name}: MP SKIP (유효 {int(valid_mp.sum())}행)')
            continue

        loglik_grp = df.loc[mask, 'gmm_log_likelihood'].values
        weak_y = weak_label_from_gmm(loglik_grp, WEAK_LABEL_QUANTILE)

        mp_best, mp_grid = grid_select_by_f1(
            mp_vals, weak_y, MP_TOP_PCT_GRID
        )
        thr_mp = mp_best['threshold']
        mp_flag_local = np.zeros(len(mp_vals), dtype=np.int8)
        mp_flag_local[valid_mp & (mp_vals >= thr_mp)] = 1

        idx = df.index[mask]
        df.loc[idx, 'mp_flag'] = mp_flag_local

        n_mp_anom = int(mp_flag_local.sum())
        n_mp_total = int(mask.sum())
        print(f"  {type_name}: top_pct={mp_best['top_pct']:.3f} "
              f"(F1={mp_best['f1']:.3f}), "
              f"이상 {n_mp_anom:,}행 ({n_mp_anom/n_mp_total*100:.2f}%)")

        if type_name in grid_records:
            grid_records[type_name]['mp_grid'] = mp_grid
            grid_records[type_name]['mp_best'] = mp_best

        # summary_rows 업데이트
        for row in summary_rows:
            if row['type'] == type_name:
                row['mp_top_pct'] = mp_best['top_pct']
                row['mp_f1'] = mp_best['f1']
                row['mp_anomalies'] = n_mp_anom
                row['mp_anomaly_pct'] = n_mp_anom / n_mp_total * 100

        plot_score_dist(
            mp_vals[valid_mp],
            f'{type_name} | Matrix Profile score',
            os.path.join(FIG_DIR, f'phase05_mp_score_{type_name}.png'),
            thr=thr_mp,
        )

    # ── 통계 기반 보조 탐지 (전체 데이터 weak label로 그리드 탐색) ──
    df, stat_grid = stat_anomaly_flags(df, full_weak)

    # ── 앙상블: 3개(IF+AE+MP) 중 2개 이상 = 고신뢰 ──
    print('\n[앙상블 (IF + AE + MP 다수결)]')
    if_flag = df['if_flag'].astype(int).values
    ae_flag = df['ae_flag'].astype(int).values
    mp_flag = df['mp_flag'].astype(int).values
    stat_flag = (
        (df['stat_zscore_flag'].astype(int).values == 1)
        | (df['stat_iqr_flag'].astype(int).values == 1)
    ).astype(int)

    vote = if_flag + ae_flag + mp_flag  # 0~3

    # 신뢰도: 2+표=high, 1표=low, 0표+통계만=stat_only
    confidence = np.full(len(df), 'normal', dtype=object)
    confidence[vote >= 2] = 'high'
    confidence[vote == 1] = 'low'
    only_stat = (vote == 0) & (stat_flag == 1)
    confidence[only_stat] = 'stat_only'

    df['anomaly_confidence'] = pd.Categorical(
        confidence,
        categories=['normal', 'stat_only', 'low', 'high'],
        ordered=True,
    )
    df['is_anomaly'] = (df['anomaly_confidence'] != 'normal').astype(np.int8)

    counts = df['anomaly_confidence'].value_counts().to_dict()
    for k in ['normal', 'stat_only', 'low', 'high']:
        v = int(counts.get(k, 0))
        print(f'  {k}: {v:,}행 ({v/len(df)*100:.2f}%)')
    log_step(
        f"앙상블 완료 (IF+AE+MP 다수결): "
        f"high={int(counts.get('high',0)):,}, "
        f"low={int(counts.get('low',0)):,}, "
        f"stat_only={int(counts.get('stat_only',0)):,}"
    )

    # ── 시각화: 앙상블 분포 ──
    plot_ensemble_overlap(
        df, os.path.join(FIG_DIR, 'phase05_ensemble_overlap.png')
    )

    # ── 저장 ──
    print('\n[저장]')

    # 1) 전체 피처 + 이상 결과 (모든 행 포함)
    full_path = f'{PROCESSED_DIR}/features.parquet'
    df.to_parquet(full_path, index=False)
    full_size = os.path.getsize(full_path) / (1024 ** 2)
    print(f'  전체 피처+결과: {full_path} ({full_size:.1f} MB)')

    # 2) 이상 탐지 결과 슬림 테이블 (분석플랜 5-6 산출물)
    slim_cols = [
        FACILITY_COL, DATE_COL, '종별', '지사',
        '총사용량',
        'gmm_log_likelihood', 'context_zscore',
        'if_score', 'if_flag',
        'ae_score', 'ae_flag',
        'mp_score', 'mp_flag',
        'stat_zscore_flag', 'stat_iqr_flag',
        'anomaly_confidence', 'is_anomaly',
    ]
    slim_cols = [c for c in slim_cols if c in df.columns]
    anomaly_only = df.loc[df['is_anomaly'] == 1, slim_cols]
    anomaly_path = f'{PROCESSED_DIR}/anomaly_results.parquet'
    anomaly_only.to_parquet(anomaly_path, index=False)
    anom_size = os.path.getsize(anomaly_path) / (1024 ** 2)
    print(f'  이상 결과 테이블: {anomaly_path} '
          f'({len(anomaly_only):,}행, {anom_size:.1f} MB)')

    # 3) 종별별 요약 CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(MODEL_DIR, 'phase5_summary.csv')
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print(f'  종별 요약: {summary_csv}')

    # 4) Phase 5 요약 JSON (그리드 탐색 전 과정 기록)
    summary = {
        'input_rows': int(len(df)),
        'input_cols': int(len(df.columns)),
        'new_columns': [
            'if_score', 'if_flag',
            'ae_score', 'ae_flag',
            'mp_score', 'mp_flag',
            'stat_zscore_flag', 'stat_iqr_flag',
            'anomaly_confidence', 'is_anomaly',
        ],
        'design': {
            'scaler': 'RobustScaler (median/IQR)',
            'scaler_rationale': (
                '입력 피처(총사용량, log-likelihood 등)가 heavy-tailed이라 '
                'StandardScaler는 극단값에 의해 일반 행이 0 근처로 압축됨. '
                'IsolationForest는 트리 기반이라 스케일에 강건하나 명목 '
                '정규화를 robust로 유지하여 피처 비교성 확보.'
            ),
            'threshold_selection': (
                'GMM log-likelihood 하위 q%를 weak label로 사용하여 '
                '각 후보 임계의 F1을 비교, 최대값 선정. GMM은 본 프로젝트의 '
                '정상 패턴 프로토타입이므로 도메인적으로 합리적인 reference.'
            ),
            'weak_label_quantile': WEAK_LABEL_QUANTILE,
        },
        'grids': {
            'if_contamination_grid': IF_CONTAMINATION_GRID,
            'ae_top_pct_grid': AE_TOP_PCT_GRID,
            'mp_top_pct_grid': MP_TOP_PCT_GRID,
            'stat_top_pct_grid': STAT_TOP_PCT_GRID,
        },
        'fixed_params': {
            'if_n_estimators': IF_N_ESTIMATORS,
            'if_max_samples': IF_MAX_SAMPLES,
            'ae_hidden': AE_HIDDEN,
            'ae_epochs': AE_EPOCHS,
            'mp_window': MP_WINDOW,
            'mp_min_days': MP_MIN_DAYS,
        },
        'per_type': summary_rows,
        'per_type_grids': grid_records,
        'stat_grid': stat_grid,
        'ensemble': {
            k: int(counts.get(k, 0))
            for k in ['normal', 'stat_only', 'low', 'high']
        },
        'log': log,
    }
    summary_path = f'{PROCESSED_DIR}/phase5_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  요약: {summary_path}')

    # ── 최종 요약 ──
    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Phase 5 완료! ({elapsed:.1f}초)')
    print(f'{"=" * 60}')
    for entry in log:
        print(f'  {entry}')
    print(f'\n  출력: {full_path} ({full_size:.1f} MB)')
    print(f'  이상 결과: {anomaly_path} ({len(anomaly_only):,}행)')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    run()
