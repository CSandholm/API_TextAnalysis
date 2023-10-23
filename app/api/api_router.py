from ai.sv_sentiment_handler import SentimentHandler
from app.api.api_utils.api_endpoints import ApiEndpoints
from app.api.models.requests import *
from app.api.models.responses import *
from fastapi import APIRouter, HTTPException

router = APIRouter()
end = ApiEndpoints("app/configs/config_endpoints.json")


@router.post(end.SV_SENTIMENT, response_model=SvSentimentResponse)
async def sv_sentiment(request_data: Request):
    try:
        sentiment_handler = SentimentHandler(request_data.input)
        sentiment = sentiment_handler.get_sentiment()
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.EN_SENTIMENT, response_model=EnSentimentResponse)
async def en_sentiment(request_data: Request):
    try:
        return None
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
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.DETECT_LANGUAGE, response_model=DetectLanguageResponse)
async def detect_language(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
