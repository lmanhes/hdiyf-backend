import os
from dotenv import load_dotenv
load_dotenv()


# app settings
PROJECT_TITLE = "Hdiyf api"
API_PREFIX = "/api"
DOCS_URL = API_PREFIX + "/docs"

# sqlalchemy settings
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DATABASE_URL = os.getenv("DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"
)