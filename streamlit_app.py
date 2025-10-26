"""
패션 브랜드 구매 의향 SSR 실험 - Streamlit 대시보드
설문 기반 합성 페르소나 213명의 3개 패션 브랜드 구매 의도 분석

- OpenAI GPT-4로 실제 응답 생성
- OpenAI Embeddings로 임베딩
- SSR로 Likert 확률분포 변환
- 실제 설문 결과와 비교 분석
"""

import streamlit as st
import pandas as pd
import polars as po
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

# 프로젝트 모듈 임포트
from semantic_similarity_rating import ResponseRater
from openai_embeddings_helper import (
    create_anchors_with_openai_embeddings,
    encode_responses_with_openai
)
from llm_response_generator import generate_batch_responses


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="한국 K-SSR 실험",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# 데이터 로딩 함수
# ============================================================================

@st.cache_data
def load_products(file_path: str = "K_Products/fashion_brands_3items.xlsx") -> pd.DataFrame:
    """패션 브랜드 데이터 로드"""
    try:
        df = pd.read_excel(file_path)

        # 컬럼명 표준화: Excel 파일의 컬럼명을 코드에서 사용하는 형식으로 매핑
        column_mapping = {
            'ProductID': 'item_id',
            'BrandName': 'brand',
            'BrandName_EN': 'product_name',
            'Category': 'category',
            'Description': 'description_korean',
            'Price_Range': 'price_range'
        }

        # 매핑이 필요한 컬럼만 변경
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # price_krw 컬럼 생성
        if 'price_krw' not in df.columns:
            if 'price_range' in df.columns:
                df['price_krw'] = 150000  # 기본값
            else:
                df['price_krw'] = 100000  # 기본값

        return df
    except Exception as e:
        st.error(f"브랜드 데이터 로드 실패: {e}")
        return pd.DataFrame()


@st.cache_data
def load_actual_survey() -> pd.DataFrame:
    """실제 설문 결과 로드 (Survey_Based_Persona_213.xlsx에 포함)"""
    try:
        df = pd.read_excel("Persona/Survey_Based_Persona_213.xlsx")

        # ID를 persona_id로 변경
        if 'ID' in df.columns:
            df['persona_id'] = df['ID']

        # 평점 문자열을 숫자로 변환
        def extract_rating(rating_str):
            if pd.isna(rating_str):
                return None
            # "1 전혀 구매 의향이 없다" -> 1
            try:
                return int(str(rating_str).split()[0])
            except:
                return None

        # 평점 컬럼들을 숫자로 변환
        if 'TINT_Rating' in df.columns:
            df['TINT_Rating'] = df['TINT_Rating'].apply(extract_rating)
        if 'MUSINSA_Rating' in df.columns:
            df['MUSINSA_Rating'] = df['MUSINSA_Rating'].apply(extract_rating)
        if 'POLO_Rating' in df.columns:
            df['POLO_Rating'] = df['POLO_Rating'].apply(extract_rating)

        return df
    except Exception as e:
        st.error(f"실제 설문 데이터 로드 실패: {e}")
        return pd.DataFrame()


@st.cache_data
def load_personas(file_path: str = "Persona/Survey_Based_Persona_213.xlsx") -> pd.DataFrame:
    """설문 기반 페르소나 데이터 로드 (213명)"""
    try:
        df = pd.read_excel(file_path)

        # 컬럼명 표준화
        column_mapping = {
            'ID': 'persona_id',
            'Gender': 'gender',
            'Age': 'age',
            'Region': 'region',
            'Economic_Status': 'economic_status',
            'Fashion_Interest': 'fashion_interest',
            'Shopping_Style': 'shopping_style',
            'Prompt_Persona': 'prompt_persona_description'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # 필수 컬럼이 없으면 기본값 생성
        if 'persona_id' not in df.columns:
            if 'ID' not in df.columns:
                df['persona_id'] = [f"P{i:03d}" for i in range(1, len(df) + 1)]

        if 'community' not in df.columns:
            if 'region' in df.columns:
                df['community'] = df['region']
            else:
                df['community'] = "General"

        if 'name' not in df.columns:
            df['name'] = df['persona_id']

        if 'occupation' not in df.columns:
            df['occupation'] = "미지정"

        # age가 "30-39세" 같은 형식이면 중간값으로 변환
        if 'age' in df.columns and df['age'].dtype == 'object':
            def parse_age(age_str):
                if pd.isna(age_str):
                    return 30
                age_str = str(age_str)
                if '-' in age_str:
                    parts = age_str.replace('세', '').split('-')
                    try:
                        return (int(parts[0]) + int(parts[1])) / 2
                    except:
                        return 30
                else:
                    try:
                        return int(age_str.replace('세', ''))
                    except:
                        return 30
            df['age'] = df['age'].apply(parse_age)

        # gender 표준화
        if 'gender' in df.columns:
            df['gender'] = df['gender'].replace({'남성': 'M', '여성': 'F', '남': 'M', '여': 'F'})

        return df
    except Exception as e:
        st.error(f"페르소나 데이터 로드 실패: {e}")
        return pd.DataFrame()


# ============================================================================
# 세션 상태 초기화 (OpenAI 임베딩 사용)
# ============================================================================

if 'rater' not in st.session_state:
    korean_anchors = [
        "전혀 구매 의향이 없다",
        "아마도 구매하지 않을 것 같다",
        "잘 모르겠다 / 보통이다",
        "구매할 가능성이 있다",
        "매우 구매 의향이 높다"
    ]

    with st.spinner("OpenAI로 앵커 임베딩 생성 중..."):
        df_anchors = create_anchors_with_openai_embeddings(
            anchor_sentences=korean_anchors,
            anchor_id="korean",
            model="text-embedding-3-small"
        )
        st.session_state.rater = ResponseRater(df_anchors)
        st.success("✓ ResponseRater 초기화 완료!")

if 'products_df' not in st.session_state:
    st.session_state.products_df = load_products()

if 'personas_df' not in st.session_state:
    st.session_state.personas_df = load_personas()

if 'actual_survey_df' not in st.session_state:
    st.session_state.actual_survey_df = load_actual_survey()

if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None


# ============================================================================
# 사이드바 - 네비게이션
# ============================================================================

with st.sidebar:
    st.title("👔 패션 브랜드 SSR")
    st.markdown("**3개 패션 브랜드 구매 의도 예측**")
    st.markdown("---")

    page = st.radio(
        "메뉴",
        ["🏠 홈", "🔬 실험 실행", "📊 결과 분석"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("**Semantic Similarity Rating**")
    st.caption("설문 213명 | 브랜드 3개")


# ============================================================================
# 유틸리티 함수
# ============================================================================

def create_community_pmf_chart(pmf_data: dict, title: str = "커뮤니티별 구매 의도 분포"):
    """PMF 분포 바차트 생성"""
    communities = list(pmf_data.keys())

    fig = go.Figure()

    colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#95E1D3', '#6BCB77']

    for i, likert in enumerate([1, 2, 3, 4, 5], 1):
        values = [pmf_data[comm]['pmf'][i-1] for comm in communities]
        fig.add_trace(go.Bar(
            name=f'{i}점',
            x=communities,
            y=values,
            text=[f'{v:.1%}' for v in values],
            textposition='inside',
            marker_color=colors[i-1]
        ))

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis_title="커뮤니티",
        yaxis_title="확률",
        yaxis_tickformat='.0%',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_mean_score_chart(score_data: dict, title: str = "평균 구매 의도 점수"):
    """평균 점수 바차트"""
    labels = list(score_data.keys())
    scores = [score_data[label]['mean_score'] for label in labels]

    colors = ['#FF6B6B' if s < 2.5 else '#FFA07A' if s < 3 else '#FFD93D' if s < 3.5 else '#95E1D3' if s < 4 else '#6BCB77' for s in scores]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            text=[f'{s:.2f}' for s in scores],
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>평균 점수: %{y:.2f} / 5.0<extra></extra>'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="그룹",
        yaxis_title="평균 점수 (1~5)",
        yaxis_range=[0, 5.5],
        height=400,
        showlegend=False
    )

    return fig


def create_attainment_chart(pmf_data: dict, threshold: int = 4):
    """Attainment Rate (≥4점 비율) 차트"""
    labels = list(pmf_data.keys())
    attainment = [pmf_data[label]['pmf'][3] + pmf_data[label]['pmf'][4] for label in labels]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=attainment,
            text=[f'{a:.1%}' for a in attainment],
            textposition='outside',
            marker_color='#6BCB77',
        )
    ])

    fig.update_layout(
        title=f"구매 의향 긍정 비율 (≥{threshold}점)",
        xaxis_title="그룹",
        yaxis_title="비율",
        yaxis_tickformat='.0%',
        yaxis_range=[0, max(attainment) * 1.2] if attainment else [0, 1],
        height=400,
        showlegend=False
    )

    return fig


# ============================================================================
# 페이지 1: 홈
# ============================================================================

def page_home():
    st.title("👔 패션 브랜드 구매 의향 SSR 실험")
    st.markdown("### 설문 기반 213명 페르소나 × 3개 패션 브랜드")

    st.markdown("---")

    personas_df = st.session_state.personas_df
    products_df = st.session_state.products_df

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("페르소나 수", f"{len(personas_df)}명", help="실제 설문 기반 합성 페르소나")

    with col2:
        st.metric("브랜드 수", f"{len(products_df)}개", help="디스이즈네버댓, 무신사, 폴로")

    with col3:
        st.metric("앵커 문장", "5단계", help="구매 의향 Likert 척도")

    st.markdown("---")

    st.markdown("## 📖 프로젝트 소개")

    st.markdown("""
    **실제 패션 브랜드 설문조사 데이터를 기반으로 생성된 213명의 합성 페르소나**를 사용하여
    LLM이 얼마나 실제 인간 응답과 유사한 구매 의향을 예측할 수 있는지 검증하는 실험입니다.

    **3개 브랜드:**
    - 👕 **디스이즈네버댓** (thisisneverthat) - 스트릿 캐주얼
    - 👖 **무신사** (MUSINSA) - 베이직 실용 패션
    - 🏇 **폴로** (POLO Ralph Lauren) - 아메리칸 클래식

    **핵심 기술:**
    - 🤖 **OpenAI GPT-4**: 페르소나 특성 기반 자연어 응답 생성
    - 🔤 **OpenAI Embeddings**: 의미 벡터 추출
    - 📊 **SSR**: 자연어 → Likert 확률분포 (PMF) 변환
    - ✅ **검증**: 실제 설문 결과와 비교
    """)

    st.markdown("---")

    if not personas_df.empty:
        st.markdown("## 👥 페르소나 특성 분포")

        col_a, col_b = st.columns(2)

        with col_a:
            if 'gender' in personas_df.columns:
                gender_counts = personas_df['gender'].value_counts()
                fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="성별 분포")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if 'age' in personas_df.columns:
                if pd.api.types.is_numeric_dtype(personas_df['age']):
                    age_bins = [0, 20, 30, 40, 50, 60, 100]
                    age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                    personas_df['age_group'] = pd.cut(personas_df['age'], bins=age_bins, labels=age_labels, right=False)
                    age_counts = personas_df['age_group'].value_counts().sort_index()
                else:
                    age_counts = personas_df['age'].value_counts()
                fig = px.bar(x=age_counts.index, y=age_counts.values, title="연령대 분포")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.info("""
    👈 **왼쪽 사이드바에서 메뉴를 선택하세요:**

    - 🔬 **실험 실행**: 페르소나 선택 → GPT 응답 생성 → SSR 변환 → 결과 확인
    - 📊 **결과 분석**: SSR 예측 vs 실제 설문 결과 비교 분석
    """)


# ============================================================================
# 페이지 2: 실험 실행
# ============================================================================

def page_experiment():
    st.title("🔬 실험 실행")
    st.markdown("선택한 페르소나와 제품에 대한 SSR 분석을 수행합니다.")

    st.info("⚠️ **주의**: OpenAI API를 실제로 호출합니다. 요금이 발생할 수 있습니다.")

    personas_df = st.session_state.personas_df
    products_df = st.session_state.products_df

    if personas_df.empty or products_df.empty:
        st.error("데이터를 불러오지 못했습니다. 파일 경로를 확인해주세요.")
        return

    st.markdown("---")

    # Step 1: 페르소나 선택
    st.markdown("### 1️⃣ 페르소나 선택")

    col1, col2 = st.columns([1, 1])

    with col1:
        selection_mode = st.radio(
            "선택 방식",
            ["개별 선택", "조건 필터"],
            help="개별 선택: 특정 persona_id 선택 / 조건 필터: 커뮤니티, 연령, 성별 등으로 필터링"
        )

    with col2:
        if selection_mode == "개별 선택":
            selected_ids = st.multiselect(
                "Persona ID 선택",
                options=personas_df['persona_id'].tolist(),
                help="여러 개 선택 가능"
            )
            selected_personas_df = personas_df[personas_df['persona_id'].isin(selected_ids)]
        else:
            # 조건 필터
            communities = sorted(personas_df['community'].unique())
            sel_communities = st.multiselect("커뮤니티", communities, default=communities[:2])

            col_a, col_b = st.columns(2)
            with col_a:
                age_min = st.number_input("최소 나이", 10, 80, 20, 5)
            with col_b:
                age_max = st.number_input("최대 나이", 10, 80, 35, 5)

            genders = personas_df['gender'].unique().tolist()
            sel_genders = st.multiselect("성별", genders, default=genders)

            # 필터 적용
            filtered = personas_df[
                (personas_df['community'].isin(sel_communities)) &
                (personas_df['age'].between(age_min, age_max)) &
                (personas_df['gender'].isin(sel_genders))
            ]
            selected_personas_df = filtered

    st.markdown(f"**선택된 페르소나 수: {len(selected_personas_df)}명**")

    if not selected_personas_df.empty:
        with st.expander("선택된 페르소나 미리보기"):
            st.dataframe(
                selected_personas_df[['persona_id', 'name', 'community', 'age', 'gender', 'occupation']],
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # Step 2: 제품 선택
    st.markdown("### 2️⃣ 제품 선택")

    with st.expander("📦 전체 제품 목록 보기 (3개)", expanded=False):
        display_df = products_df[['item_id', 'category', 'brand', 'product_name', 'description_korean']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    selected_product_ids = st.multiselect(
        "평가할 제품 선택 (여러 개 선택 가능)",
        options=products_df['item_id'].tolist(),
        format_func=lambda x: f"{x} - {products_df[products_df['item_id']==x]['product_name'].values[0]} ({products_df[products_df['item_id']==x]['brand'].values[0]})",
        default=products_df['item_id'].tolist()[:3],
        help="제품 ID를 클릭하여 선택/해제할 수 있습니다."
    )

    selected_products_df = products_df[products_df['item_id'].isin(selected_product_ids)]

    st.markdown(f"**선택된 제품 수: {len(selected_products_df)}개 / 전체 {len(products_df)}개**")

    if not selected_products_df.empty:
        with st.expander("✅ 선택된 제품 상세 정보"):
            for _, prod in selected_products_df.iterrows():
                st.markdown(f"**[{prod['item_id']}] {prod['product_name']}** ({prod['brand']})")
                st.caption(f"카테고리: {prod['category']}")
                st.info(f"📝 {prod['description_korean']}")
                st.markdown("---")

    st.markdown("---")

    # Step 3: LLM 설정
    st.markdown("### 3️⃣ LLM 설정")

    col1, col2 = st.columns(2)

    with col1:
        llm_model = st.selectbox(
            "OpenAI 모델",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini: 빠르고 저렴 (권장) | gpt-4o: 더 정확하지만 비쌈"
        )

    with col2:
        llm_temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="응답의 창의성 (0=일관적, 2=창의적)"
        )

    st.markdown("---")

    # Step 4: SSR 설정
    st.markdown("### 4️⃣ SSR 설정")

    st.info("📌 SSR 파라미터는 표준값으로 고정되어 있습니다: **Temperature (T) = 0.8**, **Epsilon (ε) = 1e-6**")

    ssr_temperature = 0.8
    epsilon = 1e-6

    st.markdown("---")

    # Step 5: 실행
    st.markdown("### 5️⃣ 실험 실행")

    total_calls = len(selected_personas_df) * len(selected_products_df)
    can_run = total_calls > 0

    if total_calls > 0:
        st.warning(f"⚠️ **총 {total_calls}번의 OpenAI API 호출**이 발생합니다. (GPT: {total_calls}회 + Embeddings: {total_calls}회)")

    if st.button("▶️ 시뮬레이션 실행 (OpenAI GPT 호출)", type="primary", disabled=not can_run, use_container_width=True):
        if not can_run:
            st.error("페르소나와 제품을 모두 선택해주세요.")
        else:
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"진행 중: {current}/{total} ({progress*100:.1f}%)")

            with st.spinner(f"🤖 OpenAI GPT로 {total_calls}개 응답 생성 중..."):
                # 1단계: GPT로 실제 응답 생성
                response_results = generate_batch_responses(
                    personas_df=selected_personas_df,
                    products_df=selected_products_df,
                    model=llm_model,
                    max_tokens=150,
                    temperature=llm_temperature,
                    progress_callback=update_progress
                )

                status_text.text(f"✅ GPT 응답 생성 완료! ({len(response_results)}개)")

            with st.spinner("🔤 OpenAI로 임베딩 생성 중..."):
                # 2단계: 응답 임베딩 생성
                response_texts = [r['response_text'] for r in response_results]
                response_embeddings = encode_responses_with_openai(
                    response_texts,
                    model="text-embedding-3-small"
                )

            with st.spinner("📊 SSR 변환 중..."):
                # 3단계: SSR 변환
                results = []
                for idx, result_item in enumerate(response_results):
                    persona = result_item['persona']
                    product = result_item['product']
                    response_text = result_item['response_text']

                    pmf = st.session_state.rater.get_response_pmfs(
                        reference_set_id="korean",
                        llm_responses=response_embeddings[idx:idx+1],
                        temperature=ssr_temperature,
                        epsilon=epsilon
                    )

                    mean_score = np.sum(np.array([1, 2, 3, 4, 5]) * pmf[0])
                    attainment = pmf[0][3] + pmf[0][4]

                    results.append({
                        'persona_id': persona['persona_id'],
                        'name': persona['name'],
                        'community': persona['community'],
                        'age': persona['age'],
                        'gender': persona['gender'],
                        'product_id': product['item_id'],
                        'product_name': product['product_name'],
                        'product_price': product['price_krw'],
                        'response_text': response_text,
                        'pmf_1': pmf[0][0],
                        'pmf_2': pmf[0][1],
                        'pmf_3': pmf[0][2],
                        'pmf_4': pmf[0][3],
                        'pmf_5': pmf[0][4],
                        'mean_likert': mean_score,
                        'attainment_ge4': attainment
                    })

                results_df = pd.DataFrame(results)
                st.session_state.experiment_results = results_df

            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ 시뮬레이션 완료! ({len(results_df)}개 응답 생성 + 임베딩 + SSR 변환)")

    # 결과 표시
    if st.session_state.experiment_results is not None:
        st.markdown("---")
        st.markdown("### 📊 실험 결과")

        results_df = st.session_state.experiment_results

        # 커뮤니티별 집계
        community_summary = {}
        for comm in results_df['community'].unique():
            comm_data = results_df[results_df['community'] == comm]
            pmf_avg = comm_data[['pmf_1', 'pmf_2', 'pmf_3', 'pmf_4', 'pmf_5']].mean().values
            community_summary[comm] = {
                'mean_score': comm_data['mean_likert'].mean(),
                'pmf': pmf_avg
            }

        # 차트
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_mean_score_chart(community_summary, "커뮤니티별 평균 점수"), use_container_width=True)
        with col2:
            st.plotly_chart(create_attainment_chart(community_summary), use_container_width=True)

        st.plotly_chart(create_community_pmf_chart(community_summary, "커뮤니티별 PMF 분포"), use_container_width=True)

        # 상세 결과 테이블
        st.markdown("### 📋 상세 결과")

        # 응답 텍스트 미리보기
        with st.expander("💬 GPT 생성 응답 샘플 (처음 5개)"):
            sample_df = results_df[['name', 'community', 'product_name', 'response_text', 'mean_likert']].head(5)
            for idx, row in sample_df.iterrows():
                st.markdown(f"**{row['name']}** ({row['community']}) × **{row['product_name']}**")
                st.info(f"💬 \"{row['response_text']}\"")
                st.caption(f"→ 평균 Likert 점수: {row['mean_likert']:.2f}/5.0")
                st.markdown("---")

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # 다운로드
        col1, col2 = st.columns(2)

        with col1:
            csv_data = results_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 결과 다운로드 (CSV)",
                data=csv_data,
                file_name=f"kssr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # 실험 메타데이터
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "personas_count": len(selected_personas_df),
                "products_count": len(selected_products_df),
                "total_responses": len(results_df),
                "llm_model": llm_model,
                "llm_temperature": llm_temperature,
                "ssr_temperature": ssr_temperature,
                "epsilon": epsilon
            }
            json_data = json.dumps(metadata, indent=2, ensure_ascii=False).encode('utf-8')
            st.download_button(
                label="📄 실험 메타데이터 (JSON)",
                data=json_data,
                file_name=f"kssr_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


# ============================================================================
# 페이지 3: 결과 분석
# ============================================================================

def page_analysis():
    st.title("📊 결과 분석")
    st.markdown("실험 결과를 다양한 관점에서 분석합니다.")

    if st.session_state.experiment_results is None:
        st.warning("⚠️ 먼저 '🔬 실험 실행' 페이지에서 실험을 수행해주세요.")
        return

    results_df = st.session_state.experiment_results

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 커뮤니티 분석", "🛍️ 제품 분석", "📈 통계 분석", "✅ 실제 설문 비교"])

    with tab1:
        st.markdown("### 커뮤니티별 분석")

        community_stats = results_df.groupby('community').agg({
            'mean_likert': ['mean', 'std', 'min', 'max'],
            'attainment_ge4': 'mean'
        }).round(3)

        community_stats.columns = ['평균 점수', '표준편차', '최소', '최대', '긍정 비율 (≥4)']
        st.dataframe(community_stats, use_container_width=True)

        community_summary = {}
        for comm in results_df['community'].unique():
            comm_data = results_df[results_df['community'] == comm]
            pmf_avg = comm_data[['pmf_1', 'pmf_2', 'pmf_3', 'pmf_4', 'pmf_5']].mean().values
            community_summary[comm] = {
                'mean_score': comm_data['mean_likert'].mean(),
                'pmf': pmf_avg
            }

        st.plotly_chart(create_mean_score_chart(community_summary, "커뮤니티별 평균 점수"), use_container_width=True)
        st.plotly_chart(create_community_pmf_chart(community_summary), use_container_width=True)

    with tab2:
        st.markdown("### 제품별 분석")

        product_stats = results_df.groupby('product_name').agg({
            'mean_likert': ['mean', 'std', 'min', 'max'],
            'attainment_ge4': 'mean'
        }).round(3)

        product_stats.columns = ['평균 점수', '표준편차', '최소', '최대', '긍정 비율 (≥4)']
        st.dataframe(product_stats, use_container_width=True)

        product_summary = {}
        for prod in results_df['product_name'].unique():
            prod_data = results_df[results_df['product_name'] == prod]
            pmf_avg = prod_data[['pmf_1', 'pmf_2', 'pmf_3', 'pmf_4', 'pmf_5']].mean().values
            product_summary[prod] = {
                'mean_score': prod_data['mean_likert'].mean(),
                'pmf': pmf_avg
            }

        st.plotly_chart(create_mean_score_chart(product_summary, "제품별 평균 점수"), use_container_width=True)

    with tab3:
        st.markdown("### 통계 분석")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("전체 평균 점수", f"{results_df['mean_likert'].mean():.2f}")

        with col2:
            st.metric("전체 표준편차", f"{results_df['mean_likert'].std():.2f}")

        with col3:
            st.metric("전체 긍정 비율", f"{results_df['attainment_ge4'].mean():.1%}")

        st.markdown("---")

        st.markdown("#### 응답 분포 엔트로피")

        entropy_data = []
        for comm in results_df['community'].unique():
            comm_data = results_df[results_df['community'] == comm]
            pmf_avg = comm_data[['pmf_1', 'pmf_2', 'pmf_3', 'pmf_4', 'pmf_5']].mean().values
            entropy = -np.sum(pmf_avg * np.log(pmf_avg + 1e-10))
            entropy_data.append({'커뮤니티': comm, '엔트로피': entropy})

        fig = go.Figure(data=[
            go.Bar(
                x=[d['커뮤니티'] for d in entropy_data],
                y=[d['엔트로피'] for d in entropy_data],
                text=[f"{d['엔트로피']:.3f}" for d in entropy_data],
                textposition='outside',
                marker_color='#4A90E2'
            )
        ])

        fig.update_layout(
            title="커뮤니티별 응답 불확실성 (낮을수록 확신이 강함)",
            xaxis_title="커뮤니티",
            yaxis_title="엔트로피",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### ✅ 실제 설문 결과와 SSR 예측 비교")

        actual_survey_df = st.session_state.actual_survey_df

        if actual_survey_df.empty:
            st.warning("실제 설문 데이터를 불러오지 못했습니다.")
            return

        # SSR 결과와 실제 설문 데이터를 매핑
        brand_to_rating_col = {
            'thisisneverthat': 'TINT_Rating',
            'TINT': 'TINT_Rating',
            'MUSINSA': 'MUSINSA_Rating',
            'POLO': 'POLO_Rating',
            'Polo Ralph Lauren': 'POLO_Rating'
        }

        brand_to_reason_col = {
            'thisisneverthat': 'TINT_Reason',
            'TINT': 'TINT_Reason',
            'MUSINSA': 'MUSINSA_Reason',
            'POLO': 'POLO_Reason',
            'Polo Ralph Lauren': 'POLO_Reason'
        }

        # SSR 결과에 실제 평점 매핑
        merged_data = []
        for _, row in results_df.iterrows():
            persona_id = row['persona_id']
            product_name = row['product_name']

            actual_row = actual_survey_df[actual_survey_df['persona_id'] == persona_id]

            if not actual_row.empty:
                actual_row = actual_row.iloc[0]

                rating_col = None
                reason_col = None
                for brand_key, col in brand_to_rating_col.items():
                    if brand_key.lower() in product_name.lower() or product_name.lower() in brand_key.lower():
                        rating_col = col
                        reason_col = brand_to_reason_col[brand_key]
                        break

                if rating_col and rating_col in actual_row:
                    actual_rating = actual_row[rating_col]
                    actual_reason = actual_row.get(reason_col, "")

                    merged_data.append({
                        'persona_id': persona_id,
                        'product_name': product_name,
                        'ssr_mean': row['mean_likert'],
                        'actual_rating': actual_rating,
                        'ssr_response': row['response_text'],
                        'actual_reason': actual_reason,
                        'difference': row['mean_likert'] - actual_rating if pd.notna(actual_rating) else None
                    })

        if not merged_data:
            st.warning("SSR 결과와 실제 설문 데이터를 매칭할 수 없습니다.")
            return

        comparison_df = pd.DataFrame(merged_data)
        comparison_df = comparison_df[comparison_df['actual_rating'].notna()]

        st.markdown(f"**비교 가능한 응답 수: {len(comparison_df)}개**")

        # 통계 지표
        st.markdown("#### 📈 예측 정확도 지표")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mae = np.mean(np.abs(comparison_df['difference']))
            st.metric("MAE (평균 절대 오차)", f"{mae:.3f}")

        with col2:
            rmse = np.sqrt(np.mean(comparison_df['difference']**2))
            st.metric("RMSE", f"{rmse:.3f}")

        with col3:
            from scipy.stats import pearsonr
            if len(comparison_df) > 1:
                corr, p_value = pearsonr(comparison_df['ssr_mean'], comparison_df['actual_rating'])
                st.metric("Pearson 상관계수", f"{corr:.3f}", f"p={p_value:.4f}")
            else:
                st.metric("Pearson 상관계수", "N/A")

        with col4:
            accuracy_1point = (np.abs(comparison_df['difference']) <= 1).sum() / len(comparison_df)
            st.metric("±1점 이내 정확도", f"{accuracy_1point:.1%}")

        st.markdown("---")

        # 브랜드별 비교 차트
        st.markdown("#### 🛍️ 브랜드별 SSR vs 실제 평점 비교")

        brand_comparison = comparison_df.groupby('product_name').agg({
            'ssr_mean': 'mean',
            'actual_rating': 'mean'
        }).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='SSR 예측',
            x=brand_comparison['product_name'],
            y=brand_comparison['ssr_mean'],
            text=[f'{v:.2f}' for v in brand_comparison['ssr_mean']],
            textposition='outside',
            marker_color='#6BCB77'
        ))

        fig.add_trace(go.Bar(
            name='실제 평점',
            x=brand_comparison['product_name'],
            y=brand_comparison['actual_rating'],
            text=[f'{v:.2f}' for v in brand_comparison['actual_rating']],
            textposition='outside',
            marker_color='#4A90E2'
        ))

        fig.update_layout(
            title="브랜드별 평균 점수 비교",
            xaxis_title="브랜드",
            yaxis_title="평균 점수 (1~5)",
            yaxis_range=[0, 5.5],
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Scatter Plot
        st.markdown("#### 📊 SSR 예측 vs 실제 평점 산점도")

        fig = px.scatter(
            comparison_df,
            x='actual_rating',
            y='ssr_mean',
            color='product_name',
            title="SSR 예측 정확도 (대각선에 가까울수록 정확)",
            labels={'actual_rating': '실제 평점', 'ssr_mean': 'SSR 예측 점수'},
            hover_data=['persona_id']
        )

        fig.add_trace(go.Scatter(
            x=[1, 5],
            y=[1, 5],
            mode='lines',
            name='완벽한 예측',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            xaxis_range=[0.5, 5.5],
            yaxis_range=[0.5, 5.5],
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 오차 분포
        st.markdown("#### 📉 예측 오차 분포")

        fig = px.histogram(
            comparison_df,
            x='difference',
            nbins=20,
            title="SSR 예측 오차 분포 (음수: 과소 예측, 양수: 과대 예측)",
            labels={'difference': '오차 (SSR - 실제)', 'count': '빈도'}
        )

        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="완벽한 예측")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 개별 응답 비교
        st.markdown("#### 💬 개별 응답 비교 (LLM 응답 vs 실제 이유)")

        comparison_display = comparison_df.sort_values('difference', key=abs, ascending=False).head(10)

        st.markdown("**오차가 큰 상위 10개 응답**")

        for idx, row in comparison_display.iterrows():
            with st.expander(f"👤 {row['persona_id']} × {row['product_name']} | SSR: {row['ssr_mean']:.2f} vs 실제: {row['actual_rating']:.0f} | 오차: {row['difference']:+.2f}"):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**🤖 LLM/SSR 응답**")
                    st.info(f"{row['ssr_response']}")
                    st.caption(f"SSR 점수: {row['ssr_mean']:.2f}/5.0")

                with col_b:
                    st.markdown("**👤 실제 설문 응답**")
                    st.success(f"{row['actual_reason']}")
                    st.caption(f"실제 평점: {row['actual_rating']:.0f}/5")

        st.markdown("---")

        # 다운로드
        st.markdown("#### 📥 비교 결과 다운로드")

        csv_comparison = comparison_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="비교 결과 CSV 다운로드",
            data=csv_comparison,
            file_name=f"ssr_vs_actual_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ============================================================================
# 메인 라우팅
# ============================================================================

def main():
    if page == "🏠 홈":
        page_home()
    elif page == "🔬 실험 실행":
        page_experiment()
    elif page == "📊 결과 분석":
        page_analysis()


if __name__ == "__main__":
    main()
