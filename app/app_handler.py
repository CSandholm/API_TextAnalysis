import json
import logging

from app.api import api_router
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class AppHandler:
    def __init__(self):
        with open("app/configs/config_app.json") as f:
            config = json.load(f)
        self.host = str(config.get("host"))
        self.port = int(config.get("port"))


def create_app():
    logger.info("Create app")
    app = FastAPI()
    logger.info("Include router")
    app.include_router(api_router.router)
    logger.info("Return app")
    return app
