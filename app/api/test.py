from fastapi import APIRouter, Request, Depends
from app.models.test import SearchRequest

router = APIRouter(tags=["test"])

def get_job_rag(request: Request):
    return request.app.state.job_rag

def get_dept_rag(request: Request):
    return request.app.state.dept_rag

@router.get("/search/job")
async def job_search(query: str, service=Depends(get_job_rag)):
    return await service.asearch(query)

@router.get("/search/dept")
async def job_search(query: str, service=Depends(get_dept_rag)):
    return await service.asearch(query)