from fastapi import Request, APIRouter, Depends

from app.service.recommand_service import RecommandService

from app.models.roadmap import RecommandRequest


router = APIRouter(tags=["roadmap"])

def get_recommand_service(request: Request):
    return request.app.state.recommand_service

@router.post("/roadmap/recommand")
async def create_recommand(request: RecommandRequest, service: RecommandService = Depends(get_recommand_service)):
    return await service.get_recommendations(request.dream_job, request.dream_dept, request.preactivity)
