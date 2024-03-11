from fastapi import Depends
from typing import Annotated
from sqlalchemy.orm import Session

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = f"mysql+pymysql://admin:Pvm9ri1C9uKSiKQEXBAL@database-1.cps5u9q9xbdf.ap-northeast-2.rds.amazonaws.com:3306/minimlops"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)
session = SessionLocal()

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]