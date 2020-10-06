import os
from dotenv import load_dotenv
load_dotenv()


# app settings
PROJECT_TITLE = "Hdiyf api"
API_PREFIX = "/api"
DOCS_URL = API_PREFIX + "/docs"
DATABASE_URL = os.getenv("DATABASE_URL")