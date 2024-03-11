from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse
import traceback
from sqlalchemy import select, func, text, sql
import csv
import io

from models.news_article import NewsArticle
from models.scraped_order import ScrapedOrder
from models.preprocessed_article import PreprocessedArticle
from database.conn import db_dependency, SessionLocal
from routers import news_scraper, preprocessor

router = APIRouter()

#각 분야별 150개 씩 가져온다(총 1200 개)
def preprocessed_articles(num:int):
    result = []
    session = SessionLocal()
    for category_no in range(8):  # 0부터 7까지의 category_no
        data = (
            session.query(PreprocessedArticle.category_no, PreprocessedArticle.formatted_text)
            .filter(PreprocessedArticle.category_no == category_no)
            .limit(num)
            .all()
        )
        print(f'#{category_no} data: {data}')
        result.extend(data)
    return result
    #     category_data = [{"category_no": category_no, "formatted_text": formatted_text} for _, formatted_text in data]
    #     result.extend(category_data)

    # return result

@router.get("/scrape-and-preprocess", status_code = status.HTTP_200_OK)
async def preprocess_articles(db: db_dependency):
    await news_scraper.index.scrape_news_articles(db)
    await preprocessor.index.preprocess_articles(db)
    return {
        "status": "success",
        "message": "[Mini MLOps] GET /data_management/scrape-and-preprocess 완료되었습니다.",
    }

@router.get("/download-preprocessed-data/{id}", status_code=status.HTTP_200_OK)
async def download_csv(db: db_dependency, id: int):
    data = (db.query(PreprocessedArticle.original_article_id, PreprocessedArticle.category_no, PreprocessedArticle.formatted_text)
        .join(NewsArticle, PreprocessedArticle.original_article_id == NewsArticle.id)
        .filter(NewsArticle.scraped_order_no == id)
        .all()
    )

    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['original_article_id', 'category_no', 'formatted_text'])
    csv_writer.writerows(data)

    response = StreamingResponse(iter([csv_data.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=preprocessed_data_{id}.csv"

    return response


@router.get("/total-ordered-data", status_code=status.HTTP_200_OK)
async def read_all(
    db: db_dependency,
    skip: int = Query(0, description="Skip the first N items", ge=0),
    limit: int = Query(12, description="Limit the number of items returned", le=100),
):
    total_ordered_data = []

    total_scraped_orders = (
        db.query(ScrapedOrder.id, ScrapedOrder.start_datetime, ScrapedOrder.end_datetime)
        .order_by(ScrapedOrder.id.desc())
        .all()
    )

    for order_id, start_datetime, end_datetime in total_scraped_orders:
        preprocessed_articles_count = (
            db.query(func.count(PreprocessedArticle.id))
            .join(NewsArticle, PreprocessedArticle.original_article_id == NewsArticle.id)
            .filter(NewsArticle.scraped_order_no == order_id)
            .scalar()
        )

        ordered_data = {
            "scraped_order_no": order_id,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "preprocessed_articles_length": preprocessed_articles_count,
        }

        total_ordered_data.append(ordered_data)

    paginated_data = total_ordered_data[skip : skip + limit]

    return {
        "status": "success",
        "message": "[Mini MLOps] GET /data_management/all-data 완료되었습니다.",
        "total_ordered_data": paginated_data,
    }

@router.get("/single-group/{id}", status_code = status.HTTP_200_OK)
async def read_single(db: db_dependency, id: int):
    scraped_order = db.query(ScrapedOrder).filter(ScrapedOrder.id == id).first()
    current_group_data = (
        db.query(NewsArticle, PreprocessedArticle)
        .join(PreprocessedArticle, PreprocessedArticle.original_article_id == NewsArticle.id)
        .filter(NewsArticle.scraped_order_no == id)
        .all()
    )
    
    preprocessed_articles = []
    for news_article, preprocessed_article in current_group_data:
        preprocessed_articles.append(preprocessed_article)


    return {
        "status": "success",
        "message": "[Mini MLOps] GET /data_management/single-preprocessed-data/:id 완료되었습니다.",
        "scraped_order": scraped_order,
        "preprocessed_articles": preprocessed_articles,
        # "length": len(current_articles),
        # "start_datetime":start_datetime,
        # "end_datetime":end_datetime,
        # "data": current_articles
    }

@router.delete("/single-group/{id}", status_code=status.HTTP_200_OK)
async def read_single(db: db_dependency, id: int):
    try:
        # 부모 행 가져오기
        scraped_order = db.query(ScrapedOrder).filter(ScrapedOrder.id == id).first()

        if scraped_order:
            # 연결된 자식 행들 가져오기
            news_articles = db.query(NewsArticle).filter(NewsArticle.scraped_order_no == id).all()

            # 연결된 자식 행들 삭제
            for news_article in news_articles:
                db.query(PreprocessedArticle).filter(PreprocessedArticle.original_article_id == news_article.id).delete()
                db.delete(news_article)

            # 부모 행 삭제
            db.delete(scraped_order)
            db.commit()

            return {
                "status": "success",
                "message": f"[Mini MLOps] GET /data_management/single-group/{id} 완료되었습니다."
            }
        else:
            return {
                "status": "failure",
                "message": f"[Mini MLOps] GET /data_management/single-group/{id} 데이터가 없습니다.",
            }
    except Exception as e:
        traceback.print_exc()
        db.rollback()
        return {
            "status": "failure",
            "message": f"[Mini MLOps] GET /data_management/single-group/{id} 실패했습니다. (데이터 삭제 실패)",
        }
