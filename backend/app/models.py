from sqlalchemy import Column,String,Integer,Float,DateTime,ForeignKey,JSON,Text
from sqlalchemy.sql import func
from .db import Base
class Product(Base):
    __tablename__='products'; id=Column(String(32),primary_key=True); title=Column(String(240),nullable=False,index=True); description=Column(Text,nullable=False); category=Column(String(80),index=True); brand=Column(String(80),index=True); price=Column(Integer,index=True); colour=Column(String(40),index=True); attributes=Column(JSON,nullable=False,default=dict); tags=Column(JSON,nullable=False,default=list); image_path=Column(String(300),nullable=False)
class ProductImage(Base):
    __tablename__='product_images'; id=Column(Integer,primary_key=True,autoincrement=True); product_id=Column(String(32),ForeignKey('products.id',ondelete='CASCADE'),nullable=False,index=True); path=Column(String(300),nullable=False)
class SearchQuery(Base):
    __tablename__='search_queries'; id=Column(String(64),primary_key=True); original_query=Column(Text,nullable=False); normalized_query=Column(Text,nullable=False); parsed_attributes=Column(JSON,nullable=False,default=dict); mode=Column(String(32),nullable=False); latency_ms=Column(Float,nullable=False); created_at=Column(DateTime(timezone=True),server_default=func.now(),index=True)
class SearchEvent(Base):
    __tablename__='search_events'; id=Column(Integer,primary_key=True,autoincrement=True); query_id=Column(String(64),ForeignKey('search_queries.id',ondelete='CASCADE'),nullable=False,index=True); product_id=Column(String(32),nullable=False); rank=Column(Integer,nullable=False); score=Column(Float,nullable=False)
class IndexVersion(Base):
    __tablename__='index_versions'; id=Column(Integer,primary_key=True,autoincrement=True); version=Column(String(120),nullable=False,index=True); product_count=Column(Integer,nullable=False); dense_model=Column(String(200),nullable=False); visual_model=Column(String(200),nullable=False); build_seconds=Column(Float,nullable=False); created_at=Column(DateTime(timezone=True),server_default=func.now())
