import json
from app.api import api_router
from fastapi import FastAPI


class AppHandler:
    def __init__(self):
        with open("app/configs/config_app.json") as f:
            config = json.load(f)
        self.host = config.get("host")
        self.port = config.get("port")


    def create_app(self):
        app = FastAPI()
        app.include_router(api_router.router)
        return app
