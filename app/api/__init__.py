from fastapi import APIRouter

from app.api.routes import news

api_router = APIRouter()
api_router.include_router(news.router, tags=["news"], prefix="/news")
