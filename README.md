# 열 원격검침 이상사용 탐지

지역난방 열량계 원격검침 시간별 사용량 데이터(2021~2025, 약 1,665만 행)를 분석하여 이상사용 패턴을 탐지하고 9개 유형으로 분류하는 비지도 학습 기반 파이프라인.

## 주요 결과

| 지표 | 본 알고리즘 | 베이스라인 (전년·전전년 대비 ±30%) |
|------|:-----------:|:-------------------------:|
| Precision | 0.639 | **0.799** |
| Recall | **0.913** | 0.414 |
| F1 | **0.752** | 0.545 |
| F2 (β=2) | **0.841** | 0.458 |

- 이상의 91%를 탐지 (베이스라인 대비 **+49.9%p**)
- 총사용량 변화 없이 시간 패턴만 변화하는 패턴이탈형도 70.6% 탐지 (베이스라인 14.1%)
- 사전 경보 시스템 — 일 평균 약 152건의 점검 후보만 산출하여 현장에 통보

## 알고리즘 구조

```
[전처리] 원본 1,687만행 → 9단계 품질 파이프라인 → 1,665만행
    ↓
[피처생성] 시간대별 비율, 변동계수, 이동평균 비율 등 38개 피처
    ↓
[정상패턴] GMM으로 종별×시즌 8그룹 정상 클러스터 학습
    ↓
[이상탐지] Isolation Forest + Autoencoder + Matrix Profile + 통계(Z/IQR)
           → 4개 OR 앙상블 이상 판정
    ↓
[유형분류] 도메인 규칙 기반 9유형 분류 + SHAP 해석
    ↓
[성능평가] 합성 이상 주입 정량 평가
```

## 실행 환경

- Python 3.10+
- 필수 패키지: `pandas numpy scikit-learn torch stumpy shap matplotlib pyarrow`

```bash
pip install pandas numpy scikit-learn torch stumpy shap matplotlib pyarrow
```

## 실행 순서

```bash
python scripts/phase1_preprocess.py          # 전처리
python scripts/phase2_feature.py             # 피처 생성
python scripts/phase3_normal_pattern.py      # GMM 정상 패턴 모델링
python scripts/phase4_anomaly_detection.py   # 이상 탐지
python scripts/phase5_classification.py      # 유형 분류
python scripts/phase6_evaluation.py          # 성능 평가
```

## 프로젝트 구조

```
├── data/
│   ├── raw/              # 원본 월별 gzip CSV (60개 파일)
│   └── processed/        # features.parquet (단일 데이터셋)
├── docs/                 # 분석 문서 (보고서 초안, 도메인 정의 등)
├── models/
│   ├── gmm/              # GMM 모델 (종별×시즌 10개 pkl)
│   └── anomaly/          # IF pkl, AE pt, Scaler pkl
├── notebooks/            # EDA 노트북
├── outputs/
│   ├── figures/          # 시각화 결과 (PNG)
│   └── results/          # 평가 결과 (CSV, JSON)
├── scripts/              # 분석 파이프라인 스크립트
└── submission/           # 제출물 (코드, 근거데이터, 정의서)
```

## 이상사용 9유형

| # | 유형 | 설명 |
|:-:|------|------|
| 1 | 장기미사용후급증형 | 7일 연속 미사용 후 급증 |
| 2 | 계절역행형 | 비활성시즌 과다 사용 |
| 3 | 야간이상형 | 야간(22~06시) 비율 이상 과다 |
| 4 | 주말이상형 | 업무/공공용 주말 평일 수준 사용 |
| 5 | 급증형 | 이동평균 대비 급격한 사용량 증가 |
| 6 | 기저유량이상형 | 최소 사용량 지속적 과다 |
| 7 | 연속가동형 | 변동 극소 + 기저유량 높음 |
| 8 | 간헐사용형 | 극단적 변동계수 |
| 9 | 패턴이탈형 | GMM/AE/MP 기반 비전형 패턴 |
