from fastapi import APIRouter


router = APIRouter()


@router.get("/rand", status_code=202)
async def get_rand_news():
    pass