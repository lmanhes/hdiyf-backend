from pydantic import BaseModel


###############  News schemas  ###############

class News(BaseModel):
    id: int
    title: str
    text: str
    label: int
    subject: str

    class Config:
        orm_mode = True