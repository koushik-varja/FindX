import pytest
from backend.app.cache import RedisCache
@pytest.mark.asyncio
async def test_cache_fallback_without_server():
    c=RedisCache('redis://127.0.0.1:6399/0'); await c.connect(); assert await c.get('x') is None; assert await c.set('x',{'a':1}) is False; await c.close()
