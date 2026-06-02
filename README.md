# Capstone Team Quant

## S&P500 Intangible-Capital Valuation Gap Strategy

본 저장소는 S&P500 기업을 대상으로 **무형자본 보정 회계 팩터**가 시장의 상대가치 괴리를 설명하고, 실제 포트폴리오 성과로 이어지는지를 검증한 캡스톤 프로젝트입니다.

핵심 아이디어는 간단합니다. US GAAP에서는 R&D 지출이 대체로 즉시 비용 처리되기 때문에 혁신기업의 자기자본과 이익이 과소계상될 수 있습니다. 본 프로젝트는 R&D와 SG&A 일부를 무형투자로 재분류해 회계 팩터를 보정하고, 이 정보가 섹터 중립 PBR 괴리와 투자 성과를 설명하는지 실험했습니다.

```text
S&P500 universe
-> price + SEC fundamentals
-> intangible-capital adjusted factors
-> sector-neutral ln(PBR) prediction
-> valuation gap ranking
-> top-k long-only portfolio
-> raw return + Fama-French 5-factor alpha test
```

## Project Highlights

| 항목 | 내용 |
| --- | --- |
| 분석 대상 | 2011-2025년 사이 S&P500에 한 번이라도 포함된 기업 |
| 최종 모델 입력 | 4,146개 firm-year row, 473개 티커 |
| 모델 | Ridge 기반 MSE 회귀, RankNet pairwise ranking |
| 전략 | 예측 valuation gap 상위 5%, 10%, 20%, 30% long-only |
| 검정 | 연간/월간 수익률 t-test, bootstrap, Rank IC, Fama-French 5요인 alpha |
| 핵심 섹터 | 전체 S&P500, 바이오 및 Information Technology |

## Key Findings

전체 S&P500 범위에서는 단순 백테스트 수익률이 양호하더라도 Fama-French 5요인으로 조정한 alpha는 대부분 통계적으로 유의하지 않았습니다. 즉, 전체 시장에서는 전략 성과 상당 부분이 기존 위험요인 노출로 설명될 수 있었습니다.

반면 **바이오 및 IT 섹터**에서는 결과가 훨씬 강하게 나타났습니다. MSE와 RankNet 모두 상위 5%, 10%, 20%, 30% 포트폴리오에서 일관된 양의 FF5 alpha가 관측되었습니다.

| 대표 전략 | 연환산 HAC alpha | p-value | 해석 |
| --- | ---: | ---: | --- |
| MSE / 바이오_IT / 3년 / 상위 10% | 26.20% | 0.0013 | 무형자본 보정 gap이 강한 저평가 신호로 작동 |
| RankNet / 바이오_IT / 5년 / 상위 10% | 27.17% | 0.0033 | 순위 학습 기반 종목 선택에서도 유사한 alpha 확인 |
| MSE / 바이오_IT / 5년 / 상위 10% | 20.20% | 0.0045 | 5년 롤링에서도 통계적으로 유의 |
| RankNet / 바이오_IT / 3년 / 상위 10% | 23.33% | 0.0033 | 모델/윈도우 변화에도 방향성 유지 |

이 결과는 무형자본 집약 산업에서 전통적 PBR이 기업의 경제적 실질을 충분히 반영하지 못할 수 있으며, 회계 보정 기반 valuation gap이 투자 가능한 신호가 될 수 있음을 시사합니다.

## Methodology

### 1. Universe Construction

Wikipedia의 S&P500 현재 구성 기업과 historical changes를 결합해 연도별 S&P500 구성원을 복원했습니다. 현재 구성원에서 출발해 편입/퇴출 이력을 역방향으로 적용하여 2011-2025년 S&P500 후보군을 구성했습니다.

### 2. Data Pipeline

가격 데이터는 Yahoo Finance chart API에서 수집하고, 재무 데이터는 SEC EDGAR CompanyFacts API에서 가져왔습니다. Fama-French 5요인 검정을 위해 Kenneth French 월별 5-factor 데이터를 별도로 사용했습니다.

| 데이터 | 출처 | 주요 용도 |
| --- | --- | --- |
| S&P500 멤버십 | Wikipedia | 연도별 투자 가능 universe |
| 가격 | Yahoo Finance chart API | valuation date 가격, 백테스트 수익률 |
| 재무제표 | SEC EDGAR CompanyFacts | R&D, SG&A, 영업이익, 자기자본 등 |
| FF5 factor | Kenneth French Data Library | 위험조정 alpha 검정 |

### 3. Intangible-Capital Adjustment

본 연구는 R&D 전체와 SG&A의 30%를 무형투자로 간주했습니다.

```text
Investment_t = R&D_t + 0.3 * SG&A_t
```

무형자본은 33% 상각률을 적용해 영구재고법 방식으로 누적했습니다.

```text
IntangibleCapital_t = (1 - 0.33) * IntangibleCapital_{t-1} + Investment_t
```

이후 보정 자기자본, 보정 영업이익, 보정 회계 팩터 `m1`, `m2`, `m3`를 생성하고 섹터 중립 `ln(PBR)`을 예측 대상으로 설정했습니다.

### 4. Modeling and Portfolio Test

실험은 모델 2종, 섹터 범위 2종, 롤링 학습기간 2종의 조합으로 구성했습니다.

| 축 | 옵션 |
| --- | --- |
| 모델 | MSE Ridge, RankNet |
| 섹터 범위 | 전체 S&P500, 바이오 및 IT |
| 롤링 윈도우 | 3년, 5년 |
| 선택 비율 | 상위 5%, 10%, 20%, 30% |

예측값과 실제 섹터 중립 PBR의 차이를 valuation gap으로 정의하고, gap 상위 종목을 동일가중 long-only 포트폴리오로 구성했습니다.

## Repository Structure

```text
.
├── README.md
├── presentation.html
├── 최종_보고서.md
├── 레퍼런스 논문/
└── 캡스톤_실험/
    ├── README.md
    ├── run_pipeline.py
    ├── run_experiment.py
    ├── run_experiment_suite.py
    ├── src/
    ├── data/
    ├── experiments/
    └── reports/
```

## Main Deliverables

| 산출물 | 설명 |
| --- | --- |
| [presentation.html](presentation.html) | 최종 발표용 HTML 프레젠테이션 |
| [최종_보고서.md](최종_보고서.md) | 연구 배경, 방법론, 결과, 한계를 정리한 최종 보고서 |
| [캡스톤_실험/README.md](캡스톤_실험/README.md) | 데이터 파이프라인 및 실험 실행 가이드 |
| [캡스톤_실험/reports/capstone_experiment_report.md](캡스톤_실험/reports/capstone_experiment_report.md) | 정량 실험 결과 보고서 |
| [레퍼런스 논문](레퍼런스%20논문) | Fama-French, 무형자본, RankNet 관련 참고 논문 |

## Quick Start

```bash
cd 캡스톤_실험
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

데이터 파이프라인 전체 실행:

```bash
.venv/bin/python run_pipeline.py --steps 01234
```

데이터 준비 상태 검증:

```bash
.venv/bin/python validate_dataset.py
```

단일 실험 실행:

```bash
.venv/bin/python run_experiment.py \
  --name sp500_mse_all_w5 \
  --model mse \
  --sector-scope all \
  --window 5
```

전체 8개 실험 조합 실행:

```bash
.venv/bin/python run_experiment_suite.py
```

## Important Notes

본 프로젝트는 학술적 실험 및 캡스톤 연구 목적의 저장소입니다. 결과는 과거 데이터 기반 백테스트이며 실제 투자 수익을 보장하지 않습니다.

현재 실험에는 다음 한계가 남아 있습니다.

- 과거 티커, 인수합병, 상폐 등으로 인한 일부 가격 데이터 결측
- 거래비용, 슬리피지, 세금 미반영
- RankNet 단일 시드 기반 실험
- 일부 섹터 정보 Unknown 처리
- 추가 placebo, 하위기간, 다중검정 보정 필요

따라서 본 결과는 캡스톤 연구의 핵심 실증 결과로는 의미가 있지만, 논문 투고 또는 실거래 전략으로 확장하려면 추가 강건성 검정이 필요합니다.

## References

주요 선행연구는 `레퍼런스 논문/`에 정리되어 있습니다.

- Fama & French (1992), *The Cross-Section of Expected Stock Returns*
- Fama & French (1993), *Common Risk Factors in the Returns on Stocks and Bonds*
- Fama & French (2015), *A Five-Factor Asset Pricing Model*
- Hulten & Hao (2008), *What Is a Company Really Worth?*
- Peters & Taylor (2017), *Intangible Capital and the Investment-q Relation*
- Burges et al. (2005), *Learning to Rank Using Gradient Descent*
- Poh, Lim, Zohren & Roberts (2021), *Building Cross-Sectional Systematic Strategies by Learning to Rank*
