from fastapi import APIRouter, HTTPException, status, Request
import traceback
from pydantic import BaseModel

from models.news_article import NewsArticle
from database.conn import db_dependency

router = APIRouter()

class Article(BaseModel):
    content: str

@router.post("/classify", status_code = status.HTTP_200_OK)
async def classify_article(article: Article, db: db_dependency):
    try:
        print('\n\033[36m[Mini MLOps] \033[32m다음은 입력된 기사입니다.')
        print('req: ', article.content)
        return {"category": 'society', "message": "[Mini MLOps] 뉴스 기사를 분류했습니다."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# async def classify_article(db: db_dependency):
#     news_articles = db.query(NewsArticle).all()
#     return {"data": news_articles, "message": "[Mini MLOps] 뉴스 기사를 불러왔습니다."}

