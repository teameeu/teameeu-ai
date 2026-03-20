import re
import json
from typing import Dict

def _fix_missing_commas_between_keys(s: str) -> str:
    return re.sub(
        r'(".*?")\s*\n\s*(")',
        r'\1,\n\2',
        s,
        flags=re.DOTALL
    )

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', s)

def clean_space_string(prompt:str) -> str:
    prompt = prompt.strip()
    prompt = re.sub(r'\s{2,}', ' ', prompt)
    prompt = prompt.replace('\n', ' ')
    prompt = prompt.replace('\t', ' ')
    return prompt

def clean_question_intent(llm_output :Dict) -> Dict:
    if 'question_intent' in llm_output:
        if llm_output['question_intent'].strip().startswith('면접관은'):
            text = llm_output['question_intent'].strip()
            if text.startswith('면접관은'):
                text = text[len('면접관은'):].lstrip()
            llm_output['question_intent'] = text
    return llm_output
                
def parse_str_to_json(text: str) -> Dict:
    """
    GPT 출력 문자열에서 JSON만 안전하게 추출하여 파싱합니다.
    우선순위:
    1. ```json 코드블록
    2. 전체 문자열에서 첫 { ~ 마지막 } 추출
    """
    # ```json 코드블록 우선 시도
    codeblock_match = re.search(
        r"```json\s*(\{.*?\})\s*```",
        text,
        re.DOTALL
    )
    if codeblock_match:
        json_str = codeblock_match.group(1)
    else:
        # 중괄호 기반 fallback
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start >= end:
            raise ValueError("JSON 객체를 찾을 수 없습니다.", text)
        text = text[start:end + 1]
        json_str = text.replace('다요.', '다.').replace('요요.', '요.').replace('\r\n', '\n')
    # JSON 파싱
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            string_pattern = r'"(?:\\.|[^"\\])*"'

            def escape_newlines(match):
                s = match.group(0)
                # 문자열 내부의 실제 개행만 escape
                return s.replace('\n', '\\n')
            fixed_json_str = re.sub(string_pattern, escape_newlines, json_str)
            fixed_json_str = _fix_missing_commas_between_keys(fixed_json_str)
            fixed_json_str = _remove_trailing_commas(fixed_json_str)
            return json.loads(fixed_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {e}\n\n추출된 JSON:\n{json_str}")
