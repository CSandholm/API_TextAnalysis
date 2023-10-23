from api_utils.endpoints import Endpoints
from enum import Enum
from models.requests import *
from models.responses import *
from fastapi import APIRouter, HTTPException

router = APIRouter()
endpoints = Endpoints(Enum)


@router.post(endpoints.SV_SENTIMENT, response_model=Response)
def sv_sentiment(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(endpoints.EN_SENTIMENT, response_model=Response)
def en_sentiment(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(endpoints.TOPIC, response_model=Response)
def topic(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(endpoints.TRANSLATION, response_model=Response)
def translation(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(endpoints.SUMMARIZE, response_model=Response)
def summarize(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(endpoints.DETECT_LANGUAGE, response_model=Response)
def detect_language(request_data: Request):
    try:
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
