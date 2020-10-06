from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.database import models, schemas

router = APIRouter()


@router.get("/rand", status_code=202, response_model=schemas.News)
async def get_rand_news(db: Session = Depends(get_db)):
    return db.query(models.News).order_by(func.random())[0]