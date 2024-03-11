from fastapi import APIRouter, HTTPException, status
import traceback
import re
# from kobert_tokenizer import KoBERTTokenizer

from models.news_article import NewsArticle
from models.preprocessed_article import PreprocessedArticle
from database.conn import db_dependency
from .category_label import category_label

from routers import news_scraper

router = APIRouter()
# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

@router.get("/preprocess", status_code = status.HTTP_200_OK)
async def preprocess_articles(db: db_dependency):
    last_scraped_order = (
        db.query(NewsArticle.scraped_order_no)
        .order_by(NewsArticle.scraped_order_no.desc())
        .limit(1)
        .first()
    )[0]
    last_scraped_news_articles = (
        db.query(NewsArticle)
        .filter(NewsArticle.scraped_order_no == last_scraped_order)
        .all()
    )
    
    print(f"\n\033[36m[Mini MLOps] \033[32m데이터 정제를 시작합니다.")
    
    non_duplicated_contents = set()
    non_duplicated_articles = []
    
    # 중복 제거
    for article in last_scraped_news_articles:
        if article.content not in non_duplicated_contents:
            non_duplicated_contents.add(article.content)
            non_duplicated_articles.append(article)
    
    # 한글 이외 단어들 제거
    preprocessed_articles_length = 0
    for article in non_duplicated_articles:
        preprocessed_article = PreprocessedArticle()
        article.title = re.sub('[^가-힣 ]', '', article.title).strip()
        article.content = re.sub('[^가-힣 ]', '', article.content).strip()
        length_of_content = len(article.content)
        
        if length_of_content > 10:
            preprocessed_articles_length += 1
            formatted_text = article.title + article.content
            preprocessed_article.category_no = category_label[article.category]
            preprocessed_article.formatted_text = formatted_text
            preprocessed_article.original_article_id = article.id
            db.add(preprocessed_article)
            db.commit()
            db.refresh(preprocessed_article)
    
    print(f"\n\033[36m[Mini MLOps] \033[32m데이터 정제가 완료되었습니다.")
        
    return {
        "status": "success",
        "message": "[Mini MLOps] 데이터 정제가 완료되었습니다.",
        "length": preprocessed_articles_length,
    }


@router.get("/scrape-and-preprocess", status_code = status.HTTP_200_OK)
async def qwefpreprocess_articles(db: db_dependency):
    await news_scraper.index.scrape_news_articles(db)
    
    last_scraped_order = (
        db.query(NewsArticle.scraped_order_no)
        .order_by(NewsArticle.scraped_order_no.desc())
        .limit(1)
        .first()
    )[0]
    last_scraped_news_articles = (
        db.query(NewsArticle)
        .filter(NewsArticle.scraped_order_no == last_scraped_order)
        .all()
    )
    
    print(f"\n\033[36m[Mini MLOps] \033[32m데이터 정제를 시작합니다.")
    
    non_duplicated_contents = set()
    non_duplicated_articles = []
    
    # 중복 제거
    for article in last_scraped_news_articles:
        if article.content not in non_duplicated_contents:
            non_duplicated_contents.add(article.content)
            non_duplicated_articles.append(article)
    
    # 한글 이외 단어들 제거
    preprocessed_articles_length = 0
    for article in non_duplicated_articles:
        preprocessed_article = PreprocessedArticle()
        article.title = re.sub('[^가-힣 ]', '', article.title).strip()
        article.content = re.sub('[^가-힣 ]', '', article.content).strip()
        length_of_content = len(article.content)
        
        if length_of_content > 10:
            preprocessed_articles_length += 1
            text = article.title + article.content
            preprocessed_article.category_no = category_label[article.category]
            preprocessed_article.formatted_text = formatted_text
            preprocessed_article.original_article_id = article.id
            preprocessed_article.original_article_id = article.id
            
            db.add(preprocessed_article)
            db.commit()
            db.refresh(preprocessed_article)
    
    print(f"\n\033[36m[Mini MLOps] \033[32m데이터 정제가 완료되었습니다.")
        
    return {
        "status": "success",
        "message": "[Mini MLOps] 데이터 정제가 완료되었습니다.",
        "length": preprocessed_articles_length,
    }
