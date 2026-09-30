from pydantic import BaseModel,Field
from typing import Literal,Any
class TextSearchRequest(BaseModel):
    query:str=Field(min_length=1,max_length=300)
    k:int=Field(default=12,ge=1,le=50)
    mode:Literal['bm25','dense','hybrid','reranked']='reranked'
    debug:bool=False
class SearchLabRequest(BaseModel):
    query:str=Field(min_length=1,max_length=300); k:int=Field(default=8,ge=1,le=20)
class ProductOut(BaseModel):
    id:str; title:str; description:str; category:str; brand:str; price:int; colour:str; attributes:dict[str,Any]; tags:list[str]; image_path:str
