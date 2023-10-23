from app_handler import AppHandler, create_app
import uvicorn

app = create_app()

if __name__ == "__main__":
    app_handler = AppHandler()
    uvicorn.run(app, host=app_handler.host, port=app_handler.port)
