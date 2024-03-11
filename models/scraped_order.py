from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.orm import relationship
from database.conn import Base

class ScrapedOrder(Base):
    __tablename__ = "scraped_orders"

    id = Column(Integer, primary_key = True, index = True)
    start_datetime = Column(DateTime)
    end_datetime = Column(DateTime)
    
    news_articles = relationship("NewsArticle", back_populates = "scraped_orders")