from __future__ import annotations
import json
class RedisCache:
    def __init__(self,url:str): self.url=url; self.client=None; self.available=False
    async def connect(self):
        try:
            import redis.asyncio as redis
            self.client=redis.from_url(self.url,decode_responses=True,socket_connect_timeout=0.35,socket_timeout=0.35)
            await self.client.ping(); self.available=True
        except Exception: self.client=None; self.available=False
    async def get(self,key:str):
        if not self.available or self.client is None:return None
        try:
            raw=await self.client.get(key); return json.loads(raw) if raw else None
        except Exception:self.available=False; return None
    async def set(self,key:str,value,ttl:int=120):
        if not self.available or self.client is None:return False
        try: await self.client.set(key,json.dumps(value,ensure_ascii=False),ex=ttl); return True
        except Exception:self.available=False; return False
    async def close(self):
        if self.client is not None:
            try:await self.client.aclose()
            except Exception:pass
