import logging
import uvicorn

from app.app_handler import create_app, AppHandler
from app.configs.config_logging.logger import config_logger

config_logger()
logger = logging.getLogger(__name__)
logger.info("Logging configured")
app = create_app()


if __name__ == "__main__":
    app_handler = AppHandler()
    logger.info(f"Host:{app_handler.host} Port:{app_handler.port}")
    uvicorn.run(app, host=app_handler.host, port=app_handler.port)
