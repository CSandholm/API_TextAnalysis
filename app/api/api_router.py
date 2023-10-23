from app.api.api_utils.api_endpoints import ApiEndpoints
from app.api.models.requests import *
from app.api.models.responses import *
from fastapi import APIRouter, HTTPException

router = APIRouter()
end = ApiEndpoints("app/configs/config_endpoints.json")


@router.post(end.SV_SENTIMENT, response_model=Response)
def sv_sentiment(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.EN_SENTIMENT, response_model=Response)
def en_sentiment(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.TOPIC, response_model=Response)
def topic(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.TRANSLATION, response_model=Response)
def translation(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.SUMMARIZE, response_model=Response)
def summarize(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.DETECT_LANGUAGE, response_model=Response)
def detect_language(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
