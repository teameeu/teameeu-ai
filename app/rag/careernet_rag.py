from app.rag.base_hybrid_rag import BaseHybridRAG

class CareernetJobHybridRAG(BaseHybridRAG):
    @property
    def data_filename(self) -> str:
        return "careernet_jobs.jsonl"

    @property
    def index_subdir(self) -> str:
        return "job"
    
class CareernetDeptHybridRAG(BaseHybridRAG):
    @property
    def data_filename(self) -> str:
        return "careernet_depts.jsonl"

    @property
    def index_subdir(self) -> str:
        return "department"