import argparse
from typing import Optional

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--max_doc_tokens', type=int, default=4096,
                    help='LLM이 생성할 최대 토큰 수')
    p.add_argument('--max_chat_tokens', type=int, default=2058,
                    help='LLM이 생성할 최대 토큰 수')
    p.add_argument('-t', '--temperature', type=float, default=.7,
                    help='답변 생성 시, 랜덤성을 결정하는 파라미터. 0 이상의 값을 가질 수 있습니다. 값이 클수록 확률적인(창의적인) 답변, 작을수록 결정적인 답변을 생성합니다. GPT-5 계열 모델은 지원하지 않습니다.')
    
    # RAG 관련
    p.add_argument('--embeddings_model', type=str, default='text-embedding-3-large',
                    help='벡터DB를 구축할때 사용할 임베딩 모델')
    p.add_argument('--chunk_size', type=int, default=1000,
                    help='벡터DB를 구축할때 사용할 청크 크기')
    p.add_argument('--chunk_overlap', type=int, default=200,
                    help='벡터DB를 구축할때 사용할 청크 오버랩 크기')
    p.add_argument('--batch_size', type=int, default=200,
                    help='벡터DB를 구축할때 사용할 배치 크기')
    p.add_argument('--max_context_length', type=int, default=500,
                    help='가져온 컨텍스트의 최대 길이(문자열 기준)')
    p.add_argument('--n_docs', type=int, default=5,
                    help='가져올 컨텍스트의 최대 개수')
    p.add_argument('--rag_data_path', type=str, default="./data/json",
                    help='RAG 구성용 데이터 경로')
    p.add_argument('--job_storage_path', type=str, default='app/faiss_db/job', 
                   help='RAG 인덱스 경로')
    p.add_argument('--department_competency_storage_path', type=str, default='app/faiss_db/department', 
                   help='학과별 인재상 매칭 데이터 경로')
    return p

# 안전하게 FastAPI에서 불러오는 함수
def load_config(args: Optional[list] = None):
    """
    FastAPI 내부에서 안전하게 config를 가져오기 위한 함수
    args=None → import 시 기본값 사용, CLI 인자는 무시
    """
    parser = define_argparser()
    if args is None:
        args = []
    return parser.parse_args(args)

if __name__ == "__main__":
    parser = define_argparser()
    config = parser.parse_args() 
    print("config:", config)