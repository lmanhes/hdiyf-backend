from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import settings



Base = declarative_base()

#engine = create_engine(settings.DATABASE_URL, connect_args={'check_same_thread': False})
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# create a database session and close it after finishing
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
