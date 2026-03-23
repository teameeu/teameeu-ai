import re
import time
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
CAREERNET_API_KEY = os.getenv("CAREERNET_API_KEY")
MAJOR_LIST_URL = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={CAREERNET_API_KEY}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&thisPage=1&perPage=1000"


def _call_get_api(SERVER_URL, session: requests.Session):
    start = time.perf_counter()
    try:
        res = session.get(SERVER_URL, timeout=120)
        latency = round(time.perf_counter() - start, 3)

        if res.status_code == 200:
            return {
                "data": json.loads(res.text.replace('"<br>','"').replace('<br>"','"').replace("<br>",", ")),
                "success": True,
                "status_code": 200,
                "latency": latency,
            }

        return {
            "data": res.json(),
            "success": False,
            "status_code": res.status_code,
            "latency": latency,
        }
    
    except Exception as e:
        return {
            "success": False,
            "status_code": getattr(res, 'status_code', None),
            "latency": round(time.perf_counter() - start, 3),
            "error": str(e)
        }


def _process_major(major):
    """개별 major 처리 함수 (워커가 실행)"""
    session = requests.Session()
    
    major_seq = major.get("majorSeq")
    represent_dept = major.get("mClass")
    field = major.get("lClass")
    relate_depts = [m.strip() for m in major.get("facilName", "").split(",") if m.strip()]

    major_detail_url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={CAREERNET_API_KEY}&svcType=api&svcCode=MAJOR_VIEW&contentType=json&gubun=univ_list&majorSeq={major_seq}"
    res = _call_get_api(major_detail_url, session)
    
    if not res.get("success"):
        return None
    
    major_detail = {}
    li_major_detail = res.get("data", {}).get("dataSearch", {}).get("content",[])
    if len(li_major_detail)>0:
        major_detail = li_major_detail[0]
    summary = major_detail.get("summary", "")
    interest = major_detail.get("interest", "")
    interests = re.split(r'[.!?]\s*', interest)
    interests = [s.strip() for s in interests if s.strip()]
    properties = major_detail.get("property", "")
    career_act = major_detail.get("career_act", {})
    universities = [m.get("schoolName") for m in major_detail.get("university", {})]
    relate_subjects = [subject for subject in major_detail.get("relate_subject", "") if subject.get("subject_description") and subject.get("subject_name")]
    
    return [
        {
            "metadata" : {
                "majorSeq": major_seq,
                "universities": universities, # 대학
                "field": field, # 계열
                "represent_dept": represent_dept, # 학과 대표명
                "summary": summary, # 학과 개요
                "relate_depts": relate_depts, # 관련 학과(유사학과)
                "relate_subjects": relate_subjects, # 관련 과목(고등학교 과목)
                "interests": interests, # 학과 인재상
                "properties": properties, # 교육목표
                "career_act": career_act, # 세특 진로 활동
                "url": major_detail_url # 출처(커리어넷)
            },
            "page_content" : f"{dept}{f'({represent_dept})' if dept != represent_dept else ''} | 인재상 | 선호하는 학생 | 학과 인재상 | {' '.join(interests)}"
        } for dept in relate_depts] + [
        {
            "metadata" : {
                "majorSeq": major_seq,
                "universities": universities, # 대학
                "field": field, # 계열
                "represent_dept": represent_dept, # 학과 대표명
                "summary": summary, # 학과 개요
                "relate_depts": relate_depts, # 관련 학과(유사학과)
                "relate_subjects": relate_subjects, # 관련 과목(고등학교 과목)
                "interests": interests, # 학과 인재상
                "properties": properties, # 교육목표
                "career_act": career_act, # 세특 진로 활동
                "url": major_detail_url # 출처(커리어넷)
            },
            "page_content" : f"{dept}{f'({represent_dept})' if dept != represent_dept else ''} | 교육목표 | {' '.join(properties)}"
        } for dept in relate_depts
    ] + [
        {
            "metadata" : {
                "majorSeq": major_seq,
                "universities": universities, # 대학
                "field": field, # 계열
                "represent_dept": represent_dept, # 학과 대표명
                "summary": summary, # 학과 개요
                "relate_depts": relate_depts, # 관련 학과(유사학과)
                "relate_subjects": relate_subjects, # 관련 과목(고등학교 과목)
                "interests": interests, # 학과 인재상
                "properties": properties, # 교육목표
                "career_act": career_act, # 세특 진로 활동
                "url": major_detail_url # 출처(커리어넷)
            },
            "page_content" : f"{dept}{f'({represent_dept})' if dept != represent_dept else ''} | 고등학교때 수강해야하는 과목 | 관련 과목 | {relate_subjects}"
        } for dept in relate_depts
    ] + [
        {
            "metadata" : {
                "majorSeq": major_seq,
                "universities": universities, # 대학
                "field": field, # 계열
                "represent_dept": represent_dept, # 학과 대표명
                "summary": summary, # 학과 개요
                "relate_depts": relate_depts, # 관련 학과(유사학과)
                "relate_subjects": relate_subjects, # 관련 과목(고등학교 과목)
                "interests": interests, # 학과 인재상
                "properties": properties, # 교육목표
                "career_act": career_act, # 세특 진로 활동
                "url": major_detail_url # 출처(커리어넷)
            },
            "page_content" : f"{dept}{f'({represent_dept})' if dept != represent_dept else ''} | 진로 활동 | 세부특기사항 활동 | 세특 추천 활동 | {career_act}"
        } for dept in relate_depts
    ]


def main():
    # 1. major_list 가져오기
    session = requests.Session()
    major_list_response = _call_get_api(MAJOR_LIST_URL, session)
    
    if not major_list_response.get("success"):
        print("학과 리스트 가져오기 실패")
        return
    
    majors = major_list_response.get("data", {}).get("dataSearch", {}).get("content", [])
    print(f"전체 학과: {len(majors)}")
    
    # 2. 병렬 처리 (워커 최대 5개)
    final_data = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 모든 작업 제출
        future_to_major = {executor.submit(_process_major, major): major for major in majors}
        
        # 완료된 작업 순서대로 처리
        for future in tqdm(as_completed(future_to_major), total=len(majors)):
            try:
                result = future.result()
                if result is not None:
                    final_data.extend(result)
                else:
                    print("오류")
            except Exception as e:
                print(f"에러 발생: {e}")
    
    # 3. 결과 저장
    print(f"처리된 학과: {len(final_data)}")
    with open("./data/json/careernet_depts.jsonl", "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()