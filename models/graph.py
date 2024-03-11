from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, BLOB
from sqlalchemy.orm import relationship
from database.conn import Base

class Graph(Base):
    __tablename__ = "graphs"

    graph_id = Column(Integer, primary_key = True, index = True)
    model_id = Column(Integer,ForeignKey("models.model_id"))
    acc_graph = Column(BLOB)
    loss_graph = Column(BLOB)
    confusion_graph = Column(BLOB)
    
    models = relationship("Model", back_populates = "graphs")
    
    