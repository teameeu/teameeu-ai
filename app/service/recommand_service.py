from app.infra.invoke_with_retry import invoke_with_retry
from app.utils.clean_str import parse_str_to_json


class RecommandService:
    def __init__(self, app_state):
        self.num_recommandaion = 5
        self.prompt_generator = app_state.prompt_generator
        self.model = app_state.model["gpt-5-mini-minimal"]
        self.dept_rag = app_state.dept_rag
        self.job_rag = app_state.job_rag

    async def get_recommendations(self, dream_job: str, dream_dept: str, preactivity: list):
        # Placeholder for recommendation logic
        # In a real implementation, this would involve complex algorithms
        # and data analysis to generate personalized recommendations for the user.

        dept_rag_results = await self.dept_rag.asearch(f"{dream_dept} 추천 활동")

        input = {
            "dream_job": dream_job,
            "dream_dept": dream_dept,
            "num_recommendation": self.num_recommandaion,
            "user_preactivity": preactivity,
            "dept_rag_results": dept_rag_results[:3]
        }
        messages = self.prompt_generator.generate_prompt("recommendation", **input)

        def check_response_format(response):
            parsed = parse_str_to_json(response.content)
            if not isinstance(parsed, dict) or "recommendations" not in parsed:
                raise ValueError("Response must be a JSON object with a 'recommendations' key")
            
            for rec in parsed["recommendations"]:
                if not isinstance(rec.get("id"), int) or not isinstance(rec.get("activity_type"), str) or not isinstance(rec.get("activity_title"), str) or not isinstance(rec.get("estimated_time"), str) or not isinstance(rec.get("detailed_todo"), list):
                    raise ValueError("Invalid recommendation format")
        
        def parse_response(response):
            return parse_str_to_json(response.content)
        

        result = invoke_with_retry(
            self.model,
            messages,
            check_fn=check_response_format,
            parse_fn=parse_response
        )
        

        return result