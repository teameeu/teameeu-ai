import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from langchain_openai import ChatOpenAI

from app.tokenizer.kiwi import KiwiTokenizer

from app.core.config import load_config
from app.core.logger import setup_logger
from app.core.middleware import RequestLogMiddleware
from app.rag.careernet_rag import CareernetJobHybridRAG, CareernetDeptHybridRAG
from app.api.health import router as health_router
from app.api.test import router as test_router

load_dotenv()
config = load_config()

# lifespan 이벤트
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.config = config
    app.state.logger = setup_logger()
    app.state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    app.state.HF_TOKEN = os.getenv("HF_TOKEN")
    app.state.kiwi = KiwiTokenizer()
    app.state.job_rag = CareernetJobHybridRAG(app)
    app.state.dept_rag = CareernetDeptHybridRAG(app)

    app.state.model = {
        "gpt-5-mini-minimal": ChatOpenAI(
            model = "gpt-5-mini",
            api_key = app.state.OPENAI_API_KEY,
            max_completion_tokens = config.max_doc_tokens,
            reasoning_effort = "minimal",
            seed = 10
        ),
        "gpt-5-mini-chat": ChatOpenAI(
            model = "gpt-5-mini",
            api_key = app.state.OPENAI_API_KEY,
            max_completion_tokens = config.max_chat_tokens,
            reasoning_effort = "minimal",
            seed = 10,
            streaming = True
        ),
        "decompose-query": ChatOpenAI(
            model = "gpt-4o-mini",
            api_key = app.state.OPENAI_API_KEY,
            max_completion_tokens = config.max_chat_tokens,
            seed = 10
        )
    }

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

app.add_middleware(RequestLogMiddleware)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router 등록
app.include_router(health_router)
app.include_router(test_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 