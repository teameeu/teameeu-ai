import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from langchain_openai import ChatOpenAI

from app.core.logger import setup_logger
from app.core.middleware import RequestLogMiddleware

load_dotenv()
# lifespan 이벤트
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    
    app.state.llm = ChatOpenAI(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    app.state.logger.info("Application startup")

    yield

    # shutdown
    app.state.logger.info("Application shutdown")


# FastAPI 앱 생성
app = FastAPI(
    title="teameeu",
    version="1.0.0",
    description="A simple FastAPI webapp",
    lifespan=lifespan
)
app.state.logger = setup_logger()
app.add_middleware(RequestLogMiddleware)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 