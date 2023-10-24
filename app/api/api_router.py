from app.api.ai.sv_sentiment_handler import SvSentimentHandler
from app.api.ai.en_sentiment_handler import EnSentimentHandler
from app.api.ai.sv_summarize_text_handler import SvSummarizeTextHandler
from app.api.api_utils.api_endpoints import ApiEndpoints
from app.api.models.requests import *
from app.api.models.responses import *
from fastapi import APIRouter, HTTPException

import asyncio

router = APIRouter()
end = ApiEndpoints("app/configs/config_endpoints.json")


@router.post(end.SV_SENTIMENT, response_model=SvSentimentResponse)
async def sv_sentiment(request_data: Request):
    try:
        sentiment_handler = SvSentimentHandler(request_data.input)
        task = asyncio.create_task(sentiment_handler.get_sentiment())
        sentiment = await task
        return SvSentimentResponse(output=sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.EN_SENTIMENT, response_model=EnSentimentResponse)
async def en_sentiment(request_data: Request):
    try:
        sentiment_handler = EnSentimentHandler(request_data.input)
        task = asyncio.create_task(sentiment_handler.get_sentiment())
        sentiment = await task
        return SvSentimentResponse(output=sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.TOPIC, response_model=TopicResponse)
async def topic(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.TRANSLATION, response_model=TranslationResponse)
async def translation(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.SUMMARIZE, response_model=SummarizeResponse)
async def summarize(request_data: Request):
    try:
        summarize_handler = SvSummarizeTextHandler(request_data.input)
        task = asyncio.create_task(summarize_handler.get_summary())
        sentiment = await task
        return SummarizeResponse(output=sentiment)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.DETECT_LANGUAGE, response_model=DetectLanguageResponse)
async def detect_language(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
