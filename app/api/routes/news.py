from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from app.database import get_db
from app.database import models, schemas
from app.utils import Type, Subject

router = APIRouter()


@router.get("/rand", status_code=202, response_model=schemas.News)
async def get_rand_news(db: Session = Depends(get_db),
                        type: Optional[Type] = None,
                        subject: Optional[Subject] = None):
    query = db.query(models.News).order_by(func.random())

    if type:
        label = 1 if type == "real" else 0
        query = query.filter(models.News.label == label)
    if subject:
        query = query.filter(models.News.subject == subject)

    return query.first()