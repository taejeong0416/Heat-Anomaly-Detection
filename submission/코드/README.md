# 열 원격검침 이상사용 탐지 - 분석 코드

## 실행 환경

- Python 3.10+
- 필수 패키지: pandas, numpy, scikit-learn, torch, stumpy, shap, matplotlib, pyarrow

```bash
pip install pandas numpy scikit-learn torch stumpy shap matplotlib pyarrow
```

## 실행 순서

```bash
python 01_전처리.py          # 원본 CSV → 정제 데이터 (PNNL 기반 9단계 품질 파이프라인)
python 02_피처생성.py        # 38개 분석 피처 생성 (시간 패턴, 추세, 통계량)
python 03_정��패턴모델링.py   # GMM 정상 패턴 클러스터링 + 그룹 내 Z-score 산출
python 04_이상탐지.py        # 4중 앙상블 이상 탐지 (IF + AE + Matrix Profile + 통계)
python 05_유형분류.py        # 9유형 규칙 기반 분류 + SHAP 해석
python 06_성���평가.py        # 합성 이상 주입 정량 평가 (vs 전년대비 ±30% 베이스라인)
```

## 입출력 데이터

| 구분 | 경로 | 설명 |
|------|------|------|
| 입력 | `data/raw/{연도}/` | 월별 gzip CSV (60개 파일, 2021~2025) |
| 중간 | `data/processed/features.parquet` | 전처리+피처+탐지 결과 누적 (단일 파일) |
| 결과 | `data/processed/anomaly_classified.parquet` | 이상 행 + 유형 분류 결과 |
| 평가 | `outputs/phase7/phase7_summary.json` | 최종 성능 지표 |

## 알고리즘 개요

```
[01 전처리] 원본 1,687만행 → 품질 검증 → 1,665만행 (98.7% 유지)
     ��
[02 피처] 시간대별 비율, 변동계수, 이동평균 비율, 자기상관 등 38개 피처
     ↓
[03 정상패턴] GMM으로 종별×시즌 8그룹 정상 클러스터 학습
              → 로그우도(정상도) + Context Z-score(그룹 내 위치) 산출
     ↓
[04 이상탐지] Isolation Forest : 다차원 고립도
              Autoencoder     : 24시간 패턴 복원 오차
              Matrix Profile  : 설비별 시계열 자기참조
              통계(Z/IQR)     : 그룹 내 극단값
              → 4개 OR 앙상블로 이상 판정
     ↓
[05 유형분류] 도메인 규칙 기반 9유형 분류 (미분류 0%)
              SHAP으로 피처 기여도 해석
     ↓
[06 성능평가] 합성 이상 주입 → Recall 0.909, F2 0.835 (Baseline 0.809 대비 우위)
```
