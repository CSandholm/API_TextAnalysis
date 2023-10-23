from app.app_handler import create_app, AppHandler
import uvicorn

app = create_app()

if __name__ == "__main__":
    app_handler = AppHandler()
    uvicorn.run(app, host=app_handler.host, port=app_handler.port)
