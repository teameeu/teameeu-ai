from pydantic import BaseModel


class PreactivityItem(BaseModel):
    id: int
    activity_type: str
    activity_title: str
    estimated_time: str
    detailed_todo: list[str]
    
    
class RecommandRequest(BaseModel):
    dream_job: str
    dream_dept: str
    preactivity: list[PreactivityItem]