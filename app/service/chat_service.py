import asyncio

from app.infra.invoke_with_retry import invoke_with_retry
from app.models.chat import RequestChat, ResponseChat


class ChatService:
    def __init__(self, app_state):
        self.job_rag = app_state.job_rag
        self.dept_rag = app_state.dept_rag
        self.query_decomposer = app_state.query_decomposer
        self.prompt_generator = app_state.prompt_generator
        self.model = app_state.model["gpt-5-mini-chat"]
        self.logger = app_state.logger

    async def _process_subquery_retrieve(self, sub_query):
        q_type = sub_query["type"]
        query = sub_query["sub_query"]

        if q_type == "job":
            results = await self.job_rag.asearch(query)
            return ("job", {"sub_query": query, "results": results[:3]})

        elif q_type == "dept":
            results = await self.dept_rag.asearch(query)
            return ("dept", {"sub_query": query, "results": results[:3]})

        else:
            return ("general", query)

    async def _run_parallel_retrieve(self, sub_queries):
        tasks = [self._process_subquery_retrieve(sq) for sq in sub_queries]

        results = await asyncio.gather(*tasks)
    
        job_queries = []
        dept_queries = []
        general_queries = []

        for q_type, result in results:
            if q_type == "job":
                job_queries.append(result)
            elif q_type == "dept":
                dept_queries.append(result)
            else:
                general_queries.append(result)

        return job_queries, dept_queries, general_queries

    
    # public method
    async def process_chat_request(self, request: RequestChat) -> ResponseChat:
        current_message = request.current_message
        conversation_history = request.conversation_history

        # decompose_query
        yield "__STATUS__질문을 분석하고 있습니다..."
        sub_queries = self.query_decomposer.decompose_query(conversation_history, current_message)
        
        yield "__STATUS__관련 정보를 검색하고 있습니다..."
        job_queries, dept_queries, general_queries =  await self._run_parallel_retrieve(sub_queries)

        # convert to langchain messages
        input = {
            "job_queries": job_queries,
            "dept_queries": dept_queries,
            "general_queries": general_queries,
            "conversation_history": conversation_history,
            "current_message": current_message
        }
        yield "__STATUS__답변을 생성하고 있습니다..."
        messages = self.prompt_generator.generate_prompt("chat", **input)
        
        async for chunk in self.model.astream(messages):
            yield chunk.content