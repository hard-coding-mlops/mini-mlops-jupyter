from sqlalchemy import Boolean, Column, Integer, DateTime, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.conn import Base
    
class PreprocessedArticle(Base):
    __tablename__ = "preprocessed_articles"
    
    id = Column(Integer, primary_key = True, index = True)
    category_no = Column(Integer)
    formatted_text = Column(Text)
    original_article_id = Column(Integer, ForeignKey("news_articles.id"))
    
    news_articles = relationship("NewsArticle", back_populates = "preprocessed_articles")