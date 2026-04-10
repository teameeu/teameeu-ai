from jinja2 import Environment, FileSystemLoader


class PromptGenerator:
    def __init__(self):
        self.txts_path = "app/prompt/txts"
        self.env = Environment(
            loader=FileSystemLoader(self.txts_path),
            autoescape=False
        )

    def generate_prompt(self, txt: str, **kwargs):
        # Read the prompt template from the text file
        template = self.env.get_template(f"{txt}.txt")
        prompts = template.render(**kwargs).split("#====")
        
        system_prompt = prompts[0].strip()
        user_prompt = prompts[1].strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]