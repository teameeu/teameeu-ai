from pydantic import BaseModel

class ConversationHistoryItem(BaseModel):
    role: str  # e.g., "user" or "assistant"
    message: str


class RequestChat(BaseModel):
    current_message: str
    conversation_history: list[ConversationHistoryItem]


class ResponseChat(BaseModel):
    role: str = "assistant"
    message: str