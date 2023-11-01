import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.api.ai.sv_sentiment_handler import SvSentimentHandler
from app.api.ai.en_sentiment_handler import EnSentimentHandler
from app.api.ai.sv_summarize_text_handler import SvSummarizeTextHandler
from app.api.ai.detect_language_handler import DetectLanguageHandler
from app.api.ai.topic_handler import TopicHandler
from app.api.ai.translation_handler import TranslationHandler
from app.api.api_utils.api_endpoints import ApiEndpoints
from app.api.models.requests import *
from app.api.models.responses import *


logger = logging.getLogger(__name__)
router = APIRouter()
end = ApiEndpoints("app/configs/config_endpoints.json")


@router.post(end.SV_SENTIMENT, response_model=SvSentimentResponse)
async def sv_sentiment(request_data: Request):
    logger.info("Sv sentiment called")
    try:
        logger.info(f"Create sv sentiment handler, input: {request_data.input}")
        sentiment_handler = SvSentimentHandler(request_data.input)
        logger.info("Creating asyncio task of get_sentiment")
        task = asyncio.create_task(sentiment_handler.get_sentiment())
        logger.info("Calling task")
        sentiment = await task
        logger.info(f"Return: {sentiment}")
        return SvSentimentResponse(output=sentiment)

    except Exception as e:
        exception = HTTPException(status_code=500, detail=str(e))
        logger.info(f"Call failed: {exception}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(end.EN_SENTIMENT, response_model=EnSentimentResponse)
async def en_sentiment(request_data: Request):
    logger.info("En sentiment called")
    try:
        logger.info(f"Create en sentiment handler, input: {request_data.input}")
        sentiment_handler = EnSentimentHandler(request_data.input)
        logger.info("Creating asyncio task of get_sentiment")
        task = asyncio.create_task(sentiment_handler.get_sentiment())
        logger.info("Calling task")
        sentiment = await task
        logger.info(f"Return: {sentiment}")
        return EnSentimentResponse(output=sentiment)

    except Exception as e:
        exception = HTTPException(status_code=500, detail=str(e))
        logger.info(f"Call failed: {exception}")
        raise exception


@router.post(end.TRANSLATION, response_model=TranslationResponse)
async def translation(request_data: TranslationRequest):
    logger.info("Translation called")
    try:
        logger.info(f"Create translation handler, input: {request_data.input}")
        translation_handler = TranslationHandler(request_data.input, request_data.src_lang, request_data.tgt_lang)
        logger.info("Creating asyncio task of get_translation")
        task = asyncio.create_task(translation_handler.get_translation())
        logger.info("Calling task")
        result = await task
        logger.info(f"Return: {result}")
        return TranslationResponse(output=result, score=0)

    except Exception as e:
        exception = HTTPException(status_code=500, detail=str(e))
        logger.info(f"Call failed: {exception}")
        raise exception


@router.post(end.SUMMARIZE, response_model=SummarizeResponse)
async def summarize(request_data: Request):
    logger.info("Summarize called")
    try:
        logger.info(f"Create summarize handler, input: {request_data.input}")
        summarize_handler = SvSummarizeTextHandler(request_data.input)
        logger.info("Creating asyncio task of get_summary")
        task = asyncio.create_task(summarize_handler.get_summary())
        logger.info("Calling task")
        summary = await task
        logger.info(f"Return: {summary}")
        return SummarizeResponse(output=summary)

    except Exception as e:
        exception = HTTPException(status_code=500, detail=str(e))
        logger.info(f"Call failed: {exception}")
        raise exception


@router.post(end.DETECT_LANGUAGE, response_model=DetectLanguageResponse)
async def detect_language(request_data: Request):
    logger.info("Detect language called")
    try:
        logger.info(f"Detect language handler, input: {request_data.input}")
        summarize_handler = DetectLanguageHandler(request_data.input)
        logger.info("Creating asyncio task of get_language")
        task = asyncio.create_task(summarize_handler.get_language())
        logger.info("Calling task")
        result = await task
        logger.info(f"Return: {result}")
        return DetectLanguageResponse(output=result)

    except Exception as e:
        exception = HTTPException(status_code=500, detail=str(e))
        logger.info(f"Call failed: {exception}")
        raise exception


@router.post(end.TOPIC, response_model=TopicResponse)
async def topic(request_data: Request):
    logger.info("Topic called")
    try:
        logger.info(f"Topic handler, input: {request_data.input}")
        topic_handler = TopicHandler(request_data.input)
        logger.info("Creating asyncio task of get_topic")
        task = asyncio.create_task(topic_handler.get_topics())
        logger.info("Calling task")
        result = await task
        logger.info(f"Return: {result}")
        return TopicResponse(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
