from fastapi import Request, APIRouter, Depends
from app.models.chat import RequestChat
from app.service.chat_service import ChatService


router = APIRouter(tags=["chat"])


def get_chat_service(request: Request):
    chat_service = request.app.state.chat_service
    return chat_service


@router.post("/chat")
async def chat(request: RequestChat, chat_service: ChatService = Depends(get_chat_service)):
    # Process the chat request
    return await chat_service.process_chat_request(request)