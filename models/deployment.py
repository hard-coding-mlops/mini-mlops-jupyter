from sqlalchemy import Column, Integer, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.conn import Base

class Deployment(Base):
    __tablename__ = "deployments"

    deploy_id = Column(Integer, primary_key = True, index = True)
    model_id = Column(Integer, ForeignKey("models.model_id"))
    deployed_at = Column(Date, default = func.current_date())
    active = Column(Boolean, default = 0)
    
    models = relationship("Model", back_populates = "deployments")