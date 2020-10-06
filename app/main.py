from fastapi import FastAPI

from app.api import api_router
import settings


def get_application():
    application = FastAPI(title=settings.PROJECT_TITLE, docs_url=settings.DOCS_URL)
    application.include_router(api_router, prefix=settings.API_PREFIX)
    return application

app = get_application()