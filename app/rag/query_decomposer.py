
from app.utils.clean_str import parse_str_to_json
from app.infra.invoke_with_retry import invoke_with_retry

class QueryDecomposer:
    def __init__(self, app):
        self.model_name = "decompose-query"
        self.model = app.state.model[self.model_name]
        self.prompt_generator=app.state.prompt_generator
    
    def _convert_to_langchain_messages(self, conversation_history, current_message: str) -> list:
        input = {"current_message": current_message, "conversation_history": conversation_history}
        langchain_messages = self.prompt_generator.generate_prompt("query_decomposer" ,**input)

        return langchain_messages
    
    def decompose_query(self, conversation_history, current_message: str) -> list:
        messages = self._convert_to_langchain_messages(conversation_history, current_message)

        # res validation func
        def check_output_format(response):
            try:
                json_res = parse_str_to_json(response.content)
                if "sub_queries" in json_res and isinstance(json_res["sub_queries"], list):
                    if "type" in json_res["sub_queries"][0] and "sub_query" in json_res["sub_queries"][0]:
                        return True
                    else:
                        return False
                else:
                    return False
            except Exception as e:
                print(f"Error parsing response: {e}")
                
                return False
        
        # res parsing func
        def return_response(response):
            json_res = parse_str_to_json(response.content)
            sub_queries = json_res.get("sub_queries", [])

            return sub_queries
        
        # invoke with retry
        sub_queries = invoke_with_retry(
            model=self.model,
            messages=messages,
            check_fn=check_output_format,
            parse_fn=return_response
        )
        
        return sub_queries
    