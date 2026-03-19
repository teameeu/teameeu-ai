from starlette.middleware.base import BaseHTTPMiddleware
import time
from fastapi import Request

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        request.app.logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)"
        )
        return response
