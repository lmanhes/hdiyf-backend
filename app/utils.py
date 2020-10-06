from enum import Enum


class Type(str, Enum):
    fake = "fake"
    real = "real"


class Subject(str, Enum):
    news = "News"
    politics = "politics"
    government_news = "Government News"
    left_news = "left-news"
    us_news = "US_News"
    middle_east_news = "Middle-east"
    politics_news = "politicsNews"
    world_news = "worldnews"