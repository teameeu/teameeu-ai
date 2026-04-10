from fastapi import Request, APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.models.chat import RequestChat
from app.service.chat_service import ChatService


router = APIRouter(tags=["chat"])


def get_chat_service(request: Request):
    chat_service = request.app.state.chat_service
    return chat_service


@router.post("/chat")
async def chat(request: RequestChat, chat_service: ChatService = Depends(get_chat_service)):
    # process the chat request
    async def event_generator():
        async for chunk in chat_service.process_chat_request(request):
            if chunk.startswith("__STATUS__"):
                msg = chunk.removeprefix("__STATUS__")
                yield f"event: status\ndata: {msg}\n\n"
            else:
                yield f"event: message\ndata: {chunk}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )