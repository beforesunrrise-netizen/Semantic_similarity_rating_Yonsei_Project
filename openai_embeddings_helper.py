"""
OpenAI 임베딩 헬퍼 - sentence-transformers 대신 OpenAI API 사용
ResponseRater를 OpenAI 임베딩으로 초기화하는 헬퍼 함수들
"""

import os
import numpy as np
import polars as po
from openai import OpenAI
from typing import List


def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    OpenAI API로 텍스트 임베딩 생성

    Parameters
    ----------
    texts : List[str]
        임베딩할 텍스트 리스트
    model : str
        OpenAI 임베딩 모델명 (기본값: text-embedding-3-small)

    Returns
    -------
    np.ndarray
        임베딩 벡터들 (shape: [n_texts, embedding_dim])
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def create_anchors_with_openai_embeddings(
    anchor_sentences: List[str],
    anchor_id: str = "korean",
    model: str = "text-embedding-3-small"
) -> po.DataFrame:
    """
    앵커 문장에 OpenAI 임베딩을 추가한 Polars DataFrame 생성

    Parameters
    ----------
    anchor_sentences : List[str]
        리커트 1~5점 앵커 문장 리스트 (5개)
    anchor_id : str
        앵커 세트 ID
    model : str
        OpenAI 임베딩 모델명

    Returns
    -------
    po.DataFrame
        임베딩이 포함된 앵커 DataFrame
    """
    if len(anchor_sentences) != 5:
        raise ValueError("앵커 문장은 정확히 5개여야 합니다 (Likert 1~5)")

    print(f"OpenAI로 앵커 임베딩 생성 중... (모델: {model})")
    embeddings = get_openai_embeddings(anchor_sentences, model=model)
    print(f"✓ 임베딩 완료! (차원: {embeddings.shape[1]})")

    df = po.DataFrame({
        "id": [anchor_id] * 5,
        "int_response": [1, 2, 3, 4, 5],
        "sentence": anchor_sentences,
        "embedding": [emb.tolist() for emb in embeddings]
    })

    return df


def encode_responses_with_openai(
    responses: List[str],
    model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    LLM 응답을 OpenAI로 임베딩 변환

    Parameters
    ----------
    responses : List[str]
        LLM 응답 텍스트 리스트
    model : str
        OpenAI 임베딩 모델명

    Returns
    -------
    np.ndarray
        응답 임베딩 (shape: [n_responses, embedding_dim])
    """
    return get_openai_embeddings(responses, model=model)
