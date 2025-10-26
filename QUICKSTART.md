# 빠른 시작 가이드

## 1분 안에 시작하기

### Step 1: 설치

```bash
# 프로젝트 폴더로 이동
cd Semantic_similarity_rating_Yonsei_Project

# 패키지 설치
pip install -r requirements.txt
```

### Step 2: (선택) OpenAI API 설정

**API 키가 있는 경우:**

```bash
# .env 파일 생성
cp .env.example .env

# 텍스트 에디터로 .env 파일 열기
# OPENAI_API_KEY=sk-your-actual-api-key-here 주석 해제하고 키 입력
```

**API 키가 없는 경우:**
- 미리 생성된 데이터로 결과 분석만 가능
- 새로운 실험은 실행 불가

### Step 3: 실행

```bash
streamlit run streamlit_app.py
```

자동으로 브라우저가 열리면서 대시보드 표시됩니다!

---

## 주요 기능

### 1. 홈 (🏠)
- 프로젝트 개요
- 데이터 통계
- 페르소나 분포 확인

### 2. 실험 실행 (🔬)
1. 페르소나 선택
   - 개별 선택: 특정 ID 선택
   - 조건 필터: 나이/성별/커뮤니티로 필터링

2. 제품 선택
   - 3개 패션 브랜드 중 선택

3. LLM 설정
   - 모델: gpt-4o-mini (권장)
   - Temperature: 0.7 (기본값)

4. 실험 실행
   - GPT 응답 생성
   - 임베딩 생성
   - SSR 변환
   - 결과 다운로드

### 3. 결과 분석 (📊)
- 커뮤니티별/제품별 분석
- 통계 지표
- 실제 설문과 비교
- 예측 정확도 평가

---

## 문제 해결

### "OPENAI_API_KEY not found"
→ `.env` 파일 생성 및 API 키 설정 필요

### "파일을 찾을 수 없습니다"
→ `Persona/` 및 `K_Products/` 폴더에 데이터 파일 확인

### 패키지 설치 오류
→ Python 3.9-3.12 버전 사용 확인

---

## 다음 단계

- 📖 [README.md](README.md) - 전체 문서
- 🔬 실제 실험 실행해보기
- 📊 결과 분석 및 시각화
- 🎓 논문 및 참고문헌 읽기

**도움이 필요하면 Issues에 질문해주세요!**
