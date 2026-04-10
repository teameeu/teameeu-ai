import asyncio

from app.models.chat import RequestChat, ResponseChat


class ChatService:
    def __init__(self, app_state):
        self.job_rag = app_state.job_rag
        self.dept_rag = app_state.dept_rag
        self.query_decomposer = app_state.query_decomposer
        self.model = app_state.model
        self.logger = app_state.logger

    async def _process_subquery_retrieve(self, sub_query):
        q_type = sub_query["type"]
        query = sub_query["sub_query"]

        if q_type == "job":
            results = await self.job_rag.asearch(query)
            return ("job", {"sub_query": query, "results": results})

        elif q_type == "dept":
            results = await self.dept_rag.asearch(query)
            return ("dept", {"sub_query": query, "results": results})

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
        sub_queries = self.query_decomposer.decompose_query(conversation_history, current_message)
        
        job_queries, dept_queries, general_queries =  await self._run_parallel_retrieve(sub_queries)
        print("job_queries:", job_queries)
        print("dept_queries:", dept_queries)
        print("general_queries:", general_queries)


        response = ResponseChat(message="This is a response to the chat request.")
        return response