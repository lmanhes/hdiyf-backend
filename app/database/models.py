from sqlalchemy import Integer, Text, Column, String

from app.database import Base


###############  News models  ###############

class News(Base):
    __tablename__ = 'news'

    id = Column(Integer, primary_key=True)
    title = Column(Text)
    text = Column(Text)
    subject = Column(String)
    label = Column(Integer)
    subject = Column(String)
