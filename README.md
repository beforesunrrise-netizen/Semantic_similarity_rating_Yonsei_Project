# Semantic Similarity Rating (SSR) - Yonsei Project

> **LLM 자연어 응답을 확률적 Likert 분포로 변환하는 통합 기술 아키텍처**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 개요

**Semantic-Similarity Rating (SSR)**은 대형 언어 모델(LLM)이 생성한 자유응답을 의미적 유사도(Semantic Similarity)에 기반해 **Likert 척도 확률분포(1~5점)**로 변환하는 방법론입니다.

### 주요 특징

- ✅ LLM의 자연어 응답을 정량적 평가로 변환
- ✅ 단일 점수가 아닌 **확률 질량 함수(PMF)** 제공
- ✅ 응답의 불확실성과 뉘앙스 보존
- ✅ 실제 설문 결과와 비교 분석 가능

---

## 🎯 주요 응용 분야

1. **소비자 리서치**: 신제품/브랜드 구매 의향 시뮬레이션
2. **커뮤니티 간 문화 비교**: 집단별 선호도 차이 분석
3. **LLM 모델 평가**: 정성 평가의 정량화
4. **초기 설문 대체**: 빠른 시장 반응 예측

---

## 🏗️ 프로젝트 구조

```
Semantic_similarity_rating_Yonsei_Project/
├── README.md                          # 프로젝트 문서
├── requirements.txt                   # Python 패키지 의존성
├── .env.example                       # 환경 변수 템플릿
├── .gitignore                         # Git 제외 파일 목록
│
├── streamlit_app.py                   # Streamlit 웹 대시보드
├── llm_response_generator.py          # OpenAI GPT 응답 생성
├── openai_embeddings_helper.py        # OpenAI 임베딩 헬퍼
│
├── semantic_similarity_rating/        # 핵심 SSR 모듈
│   ├── __init__.py
│   └── compute.py                     # PMF 계산 로직
│
├── Persona/                           # 페르소나 데이터
│   └── Survey_Based_Persona_213.xlsx  # 설문 기반 213명
│
└── K_Products/                        # 제품 데이터
    └── fashion_brands_3items.xlsx     # 패션 브랜드 3개
```

---

## 🚀 빠른 시작

### 1. 환경 설정

**필요 조건:**
- Python 3.9 이상 (3.13은 일부 패키지 미지원)
- OpenAI API Key (선택사항 - 실제 LLM 응답 생성 시 필요)

**설치:**

```bash
# 저장소 클론 또는 다운로드
cd Semantic_similarity_rating_Yonsei_Project

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. OpenAI API 설정 (선택사항)

OpenAI API를 사용하려면:

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

**⚠️ 주의:**
- API 키 없이는 실시간 LLM 응답 생성 불가
- 미리 생성된 결과 데이터로 분석은 가능
- `.env` 파일은 절대 git에 커밋하지 마세요 (.gitignore에 포함됨)

### 3. Streamlit 대시보드 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 자동으로 `http://localhost:8501` 열림

---

## 📊 사용 방법

### Streamlit 웹 인터페이스

1. **홈 페이지**
   - 프로젝트 개요 및 데이터 통계 확인
   - 페르소나 특성 분포 시각화

2. **실험 실행**
   - 페르소나 선택 (개별 선택 또는 조건 필터)
   - 제품 선택
   - LLM 모델 및 파라미터 설정
   - OpenAI GPT로 응답 생성
   - SSR 변환 및 결과 시각화

3. **결과 분석**
   - 커뮤니티별/제품별 분석
   - 통계 지표 (평균, 표준편차, 긍정 비율)
   - 실제 설문 결과와 SSR 예측 비교
   - 예측 정확도 (MAE, RMSE, Pearson 상관계수)

### Python 코드로 직접 사용

```python
from semantic_similarity_rating import ResponseRater
from openai_embeddings_helper import create_anchors_with_openai_embeddings
import pandas as pd

# 1. 앵커 문장 정의
korean_anchors = [
    "전혀 구매 의향이 없다",
    "아마도 구매하지 않을 것 같다",
    "잘 모르겠다 / 보통이다",
    "구매할 가능성이 있다",
    "매우 구매 의향이 높다"
]

# 2. ResponseRater 초기화
df_anchors = create_anchors_with_openai_embeddings(
    anchor_sentences=korean_anchors,
    anchor_id="korean"
)
rater = ResponseRater(df_anchors)

# 3. 응답 평가
test_responses = ["정말 마음에 들어요!", "별로 안 좋아요"]
from openai_embeddings_helper import encode_responses_with_openai
response_embeddings = encode_responses_with_openai(test_responses)

pmfs = rater.get_response_pmfs("korean", response_embeddings)
print(f"확률 분포:\n{pmfs}")
```

---

## 🔬 SSR 방법론

### 핵심 절차

1. **Likert 기준문장(Anchors) 정의**
   - 1~5점 각 점수를 대표하는 문장

2. **임베딩 계산**
   - 응답 문장과 앵커 문장의 의미 벡터 추출

3. **코사인 유사도 계산**
   - 응답과 각 앵커 간의 유사도 측정

4. **확률분포(PMF) 변환**
   ```
   s'_i = s_i - min(s) + ε
   p_i = exp(s'_i / T) / Σ exp(s'_j / T)
   ```
   - T: Temperature (분포의 날카로움 조절)
   - ε: Regularization (수치 안정성)

5. **기댓값 계산**
   ```
   Mean Likert = Σ i × p_i  (i=1~5)
   ```

### 검증 지표

- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE**: Root Mean Squared Error
- **Pearson Correlation**: 실제 평점과의 상관계수
- **Attainment Rate**: 긍정 응답 비율 (P[L≥4])
- **Entropy**: 응답 불확실성

---

## 📁 데이터 구성

### 페르소나 데이터

**파일**: `Persona/Survey_Based_Persona_213.xlsx`

**필수 컬럼**:
- `ID` / `persona_id`: 고유 식별자
- `Gender`: 성별
- `Age`: 나이 또는 연령대
- `Region` / `community`: 지역 또는 커뮤니티
- `Prompt_Persona`: 페르소나 설명 (LLM 프롬프트용)
- `TINT_Rating`, `MUSINSA_Rating`, `POLO_Rating`: 실제 설문 평점
- `*_Reason`: 실제 설문 응답 이유

### 제품 데이터

**파일**: `K_Products/fashion_brands_3items.xlsx`

**필수 컬럼**:
- `ProductID` / `item_id`: 제품 ID
- `BrandName` / `brand`: 브랜드명
- `Category`: 카테고리
- `Description`: 제품 설명
- `Price_Range`: 가격대

---

## 🛡️ 보안 및 주의사항

### API 키 관리

- ❌ **절대** `.env` 파일을 git에 커밋하지 마세요
- ✅ `.env.example`만 커밋하고, 실제 키는 로컬에만 보관
- ✅ API 키는 환경 변수로만 관리

### API 사용료

- OpenAI API는 **유료**입니다
- 실험 전 예상 비용 확인:
  - GPT-4o-mini: ~$0.15 / 1M input tokens (권장)
  - text-embedding-3-small: ~$0.02 / 1M tokens
- 213명 × 3제품 = 639회 호출 예상

### 데이터 프라이버시

- 실제 개인정보는 절대 저장하지 마세요
- 합성 페르소나만 사용
- 실험 결과 공유 시 민감 정보 제거

---

## 🔧 개발 가이드

### 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/test_compute.py
```

### 코드 스타일

```bash
# 코드 포맷팅
black *.py semantic_similarity_rating/

# 린트 체크
flake8 *.py
```

---

## 📚 참고문헌

1. **Maier, B. F., et al. (2025).**
   Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings.
   *arXiv preprint*.

2. **Horton, J. J. (2023).**
   Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?
   *NBER Working Paper No. 31122*.

3. **Argyle, L. P., et al. (2023).**
   Out of One, Many: Using Language Models to Simulate Human Samples.
   *Political Analysis*, 31(3), 337-351.

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 환영합니다!

---



---

## 🎓 인용

이 프로젝트를 연구에 사용하시는 경우, 다음과 같이 인용해주세요:

```bibtex
@software{ssr_yonsei_2025,
  title={Semantic Similarity Rating Framework},
  author={Yonsei University Research Team},
  year={2025},
  url={https://github.com/your-repo/semantic-similarity-rating}
}
```