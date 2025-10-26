"""
OpenAI GPT를 사용한 LLM 응답 생성 모듈
페르소나 + 제품 설명 → GPT 자연어 응답 생성
"""

import os
from openai import OpenAI
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def generate_purchase_intent_response(
    persona: pd.Series,
    product: pd.Series,
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
    temperature: float = 0.7
) -> str:
    """
    페르소나와 제품 정보를 기반으로 GPT로 구매 의향 응답 생성

    Parameters
    ----------
    persona : pd.Series
        페르소나 정보 (community, name, age, gender, prompt_persona_description 등)
    product : pd.Series
        제품 정보 (product_name, description_korean, features, price_krw 등)
    model : str
        사용할 OpenAI 모델 (기본값: gpt-4o-mini)
    max_tokens : int
        최대 응답 길이
    temperature : float
        창의성 조절 (0~2)

    Returns
    -------
    str
        자연어 구매 의향 응답 (1~2문장)
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 프롬프트 구성
    persona_desc = persona.get('prompt_persona_description', '')
    name = persona.get('name', '사용자')
    age = persona.get('age', 30)
    gender = persona.get('gender', '남')
    community = persona.get('community', '')
    linguistic_style = persona.get('linguistic_style', '')

    product_name = product.get('product_name', '')
    description = product.get('description_korean', '')
    features = product.get('features', '')
    price = product.get('price_krw', 0)

    system_prompt = f"""당신은 한국의 특정 온라인 커뮤니티({community}) 사용자입니다.

페르소나 설정:
- 이름: {name}
- 나이: {age}세
- 성별: {gender}
- 언어 스타일: {linguistic_style}
- 특징: {persona_desc}

이 페르소나의 시각과 말투로 제품에 대한 구매 의향을 자연스럽게 표현하세요.
**1~2문장**으로 짧고 자연스럽게 작성하세요."""

    user_prompt = f"""다음 제품에 대한 당신의 구매 의향을 자연스럽게 표현해주세요:

제품명: {product_name}
설명: {description}
특징: {features}
가격: {price:,}원

**중요:**
- 1~2문장으로 간결하게
- 페르소나의 언어 스타일 반영
- 점수를 직접 말하지 말고, 감정/의향을 자연스럽게 표현
- 예: "가격도 괜찮고 맛있을 것 같아서 꼭 사고 싶어요", "너무 비싸서 안 살 것 같음 ㅋㅋ"
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1
        )

        answer = response.choices[0].message.content.strip()
        # 따옴표 제거
        answer = answer.strip('"').strip("'")
        return answer

    except Exception as e:
        print(f"❌ OpenAI API 오류: {e}")
        # 폴백: 간단한 응답
        return f"제품이 괜찮아 보이네요. 가격도 적당한 편이에요."


def generate_batch_responses(
    personas_df: pd.DataFrame,
    products_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
    temperature: float = 0.7,
    progress_callback=None
) -> List[Dict]:
    """
    여러 페르소나 × 제품 조합에 대해 배치로 응답 생성

    Parameters
    ----------
    personas_df : pd.DataFrame
        선택된 페르소나 데이터프레임
    products_df : pd.DataFrame
        선택된 제품 데이터프레임
    model : str
        OpenAI 모델명
    max_tokens : int
        최대 토큰 수
    temperature : float
        Temperature
    progress_callback : callable, optional
        진행상황 콜백 함수 (current, total)

    Returns
    -------
    List[Dict]
        각 조합의 (persona, product, response_text) 딕셔너리 리스트
    """

    results = []
    total = len(personas_df) * len(products_df)
    current = 0

    for _, persona in personas_df.iterrows():
        for _, product in products_df.iterrows():
            current += 1

            if progress_callback:
                progress_callback(current, total)

            response_text = generate_purchase_intent_response(
                persona=persona,
                product=product,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            results.append({
                'persona': persona,
                'product': product,
                'response_text': response_text
            })

    return results
