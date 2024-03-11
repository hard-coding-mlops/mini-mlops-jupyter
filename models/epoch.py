from sqlalchemy import Column, Integer, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.conn import Base

class Epoch(Base):
    __tablename__ = "epochs"

    epoch_id = Column(Integer, primary_key = True, index = True)
    model_id = Column(Integer,ForeignKey("models.model_id"))
    epoch_number = Column(Integer)
    train_acc = Column(Float)
    test_acc = Column(Float)
    train_loss = Column(Float)
    test_loss = Column(Float)
    
    models = relationship("Model", back_populates = "epochs")