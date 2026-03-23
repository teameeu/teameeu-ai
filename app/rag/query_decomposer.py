
from app.utils.clean_str import parse_str_to_json

class QueryDecomposer:
    def __init__(self, app):
        self.model_name = "decompose-query"
        self.model = app.state.model[self.model_name]
        self.prompt_generator=app.state.prompt_generator
    
    def _convert_to_langchain_messages(self, conversation_history, current_message: str) -> list:
        template=self.prompt_generator.create_base_template("rag/decompose_query")
        llm_input = {"current_message": current_message, "conversation_history": conversation_history}
        langchain_messages = template.format_messages(**llm_input)
        return langchain_messages
    
    def decompose_query(self, conversation_history, current_message: str) -> list[str]:
        messages = self._convert_to_langchain_messages(conversation_history, current_message)
        response = self.model.invoke(messages)
        json_res = parse_str_to_json(response.content)
        sub_queries = json_res.get("sub_queries", [])
        return sub_queries
    