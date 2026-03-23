import re
import time
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
CAREERNET_API_KEY = os.getenv("CAREERNET_API_KEY")
JOB_LIST_URL = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={CAREERNET_API_KEY}&svcType=api&svcCode=JOB&contentType=json&gubun=job_apti_list&perPage=1000"
JOB_DETAIL_URL = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={CAREERNET_API_KEY}&svcType=api&svcCode=JOB_VIEW&contentType=json&gubun=job_apti_list&perPage=1000&jobdicSeq="


def _call_get_api(SERVER_URL, session: requests.Session):
    start = time.perf_counter()
    res = None
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

def _process_job(job):
    """개별 job 처리 함수 (워커가 실행)"""
    session = requests.Session()
    
    jobdic_seq = job.get("jobdicSeq")
    job_detail_url = f"{JOB_DETAIL_URL}{jobdic_seq}"
    res = _call_get_api(job_detail_url, session)
    
    if not res.get("success"):
        return None
    
    job_detail = {}
    li_job_detail = res.get("data", {}).get("dataSearch", {}).get("content",[])
    if len(li_job_detail)>0:
        job_detail = li_job_detail[0]
    
    # pprint(job_detail)
    job = job_detail.get("job", "") # 직업(대표명)
    division = job_detail.get("division", "")[0].get("cnet_job_dvs") # 직업 분류
    ability = job_detail.get("ability", "").replace("\n", "").replace("\r", "") if job_detail.get("ability", "") else "" # 핵심 능력
    similar_jobs = job_detail.get("similarJob", "") # 유사직업명
    similar_jobs = [s.strip() for s in similar_jobs.split(",") if s.strip()] if similar_jobs else []
    summary = job_detail.get("summary", "").replace("\n", "").replace("\r", "") if job_detail.get("summary", "") else "" # 하는 일
    aptitude = job_detail.get("aptitude", "").replace("\n", "").replace("\r", "") if job_detail.get("aptitude", "") else "" # 적성 및 흥미
    possibility = job_detail.get("job_possibility")[0].get("possibility", "").replace("\n", "").replace("\r", "") if job_detail.get("job_possibility")[0].get("possibility", "") else "" # 직업 전망
    empway = job_detail.get("stateofemp", [])[0].get("empway", "").replace("\n", "").replace("\r", "") if job_detail.get("stateofemp", [])[0].get("empway", "") else "" # 채용 방법
    employment = job_detail.get("stateofemp", [])[1].get("employment", "").replace("\n", "").replace("\r", "") if job_detail.get("stateofemp", [])[1].get("employment", "") else "" # 입적 및 취업방법
    salery = job_detail.get("stateofemp", [])[2].get("salery", "").replace("\n", "").replace("\r", "") if job_detail.get("stateofemp", [])[2].get("salery", "") else "" # 임금 정보
    capacity = job_detail.get("capacity_major", [])[0].get("capacity", "").split(",") if job_detail.get("capacity_major", [])[0].get("capacity", "") else [] # 관련 자격증
    dept = job_detail.get("capacity_major", [])[1].get("major", []) # 관련 학과
    dept = [d.get("MAJOR_NM") for d in dept]
    preparation = job_detail.get("prepareway", [])[0].get("preparation", "").replace("\n", "").replace("\r", "") if job_detail.get("prepareway", [])[0].get("preparation", "") else "" # 준비 사항
    training = job_detail.get("prepareway", [])[1].get("training", "").replace("\n", "").replace("\r", "") if job_detail.get("prepareway", [])[1].get("training", "") else "" # 훈련 과정
    certification = job_detail.get("prepareway", [])[2].get("certification", "").replace("\n", "").replace("\r", "") if job_detail.get("prepareway", [])[2].get("certification", "") else "" # 관련 자격증 간단한 설명

    dept_text = " ".join(dept)
    capacity_text = " ".join(capacity)

    return [
        {
            "metadata": {
                "job": job,
                "division": division,
                "ability": ability,
                "similar_jobs": similar_jobs,
                "summary": summary,
                "aptitude": aptitude,
                "possibility": possibility,
                "empway": empway,
                "employment": employment,
                "salery": salery,
                "capacity": capacity,
                "dept": dept,
                "preparation": preparation,
                "training": training,
                "certification": certification,
            },
            "page_content": f"{job} | 준비 | 취업 준비 | 준비 방법 | {preparation} {f' | 관련 학과: {dept_text}' if len(dept_text) > 0  else ''}{f' | 관련 자격증: {capacity_text} {certification}' if len(dept_text) > 0  else ''}{f' | 관련 교육: {training}' if len(training) > 0  else ''}".replace("  ", " ").strip()
        }, 
        {
            "metadata": {
                "job": job,
                "division": division,
                "ability": ability,
                "similar_jobs": similar_jobs,
                "summary": summary,
                "aptitude": aptitude,
                "possibility": possibility,
                "empway": empway,
                "employment": employment,
                "salery": salery,
                "capacity": capacity,
                "dept": dept,
                "preparation": preparation,
                "training": training,
                "certification": certification,
            },
            "page_content": f"{job} | 핵심 직업 역량 | 핵심 역량 | 적성 | {ability} {aptitude}".replace("  ", " ").strip()
        }, 
        {
            "metadata": {
                "job": job,
                "division": division,
                "ability": ability,
                "similar_jobs": similar_jobs,
                "summary": summary,
                "aptitude": aptitude,
                "possibility": possibility,
                "empway": empway,
                "employment": employment,
                "salery": salery,
                "capacity": capacity,
                "dept": dept,
                "preparation": preparation,
                "training": training,
                "certification": certification,
            },
            "page_content": f"{job}(분류: {division}) | 직업 정보 | 개요 | 하는 일 | {summary}{f' | 직업 전망: {possibility}' if len(possibility)>0 else ""} | {empway}".replace("  ", " ").strip()
        }, 
        {
            "metadata": {
                "job": job,
                "division": division,
                "ability": ability,
                "similar_jobs": similar_jobs,
                "summary": summary,
                "aptitude": aptitude,
                "possibility": possibility,
                "empway": empway,
                "employment": employment,
                "salery": salery,
                "capacity": capacity,
                "dept": dept,
                "preparation": preparation,
                "training": training,
                "certification": certification,
            },
            "page_content": f"{job} | 임금 | {salery}{f' | 직업 전망: {possibility}' if len(possibility)>0 else ""} {employment}".replace("  ", " ").strip()
        }, 
        ]


def main():
    # 1. jobs_list 가져오기
    session = requests.Session()
    jobs_list_response = _call_get_api(JOB_LIST_URL, session)
    
    if not jobs_list_response.get("success"):
        print("학과 리스트 가져오기 실패")
        return
    
    jobs = jobs_list_response.get("data", {}).get("dataSearch", {}).get("content", [])

    print(f"전체 직업: {len(jobs)}")
    
    # 2. 병렬 처리 (워커 최대 5개)
    final_data = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 모든 작업 제출
        future_to_major = {executor.submit(_process_job, major): major for major in jobs}
        
        # 완료된 작업 순서대로 처리
        for future in tqdm(as_completed(future_to_major), total=len(jobs)):
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
    with open("./data/json/careernet_jobs.jsonl", "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()