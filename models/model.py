from sqlalchemy import Column, Integer, Date, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.conn import Base
from .graph import Graph
from .epoch import Epoch
from .deployment import Deployment

class Model(Base):
    __tablename__ = "models"

    model_id = Column(Integer, primary_key = True, index = True)
    model_name = Column(Text)
    created_at = Column(Date, default = func.current_date())
    num_epochs = Column(Integer)    # 5
    batch_size = Column(Integer)    # 8
    max_length = Column(Integer)    # 512
    warmup_ratio = Column(Float)    # 0.1
    max_grad_norm = Column(Integer) # 1
    learning_rate = Column(Float)   # 5e-5
    split_rate = Column(Float)      # 0.25
    data_length = Column(Integer)   # 1200
    acc = Column(Float)
    loss = Column(Float)            # 41.232
    accuracy = Column(Float)        # 82.193
    precision_value = Column(Float)       # 0.0
    recall = Column(Float)          # 0.0
    f1 = Column(Float)              # 0.0
    
    graphs = relationship("Graph", back_populates = "models")
    epochs = relationship("Epoch", back_populates = "models")
    deployments = relationship("Deployment", back_populates = "models")
    clients = relationship("Client", back_populates = "models")
