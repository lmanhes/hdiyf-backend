import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

from app.database.models import News
from app.database import SessionLocal, engine, Base


db = SessionLocal()
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

df = pd.read_csv("data/clean_fake_news_dataset.csv")

for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"\n{i+1} / {len(df)} news processed\n")
    news = News(title=row.title,
               text=row.text,
               subject=row.subject,
               label=row.label)
    db.add(news)
db.commit()

