import glob
import time
import json
import fcntl
import pickle
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from app.rag.embedding import Qwen3Embeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from rank_bm25 import BM25Okapi

class HybridRAG:
    def __init__(self, app):
        self.logger = app.state.logger
        self.rag_data_path = Path(app.state.config.rag_data_path)
        self.research_storage_path = Path(app.state.config.research_storage_path)
        self.retriever: ContextualCompressionRetriever | None = None
        self.kiwi = app.state.kiwi
        self.thresholds = 0.4
        self.k = 10

        # Embeddings
        self.embeddings = Qwen3Embeddings()
        
        self._init_retriever()
        
    def _load_bm25_index(self, bm25_path):
        """BM25 Retriever 초기화 with 토크나이징 캐싱"""
        
        self.logger.info("기존 BM25 인덱스 로드 중...")
        
        with open(bm25_path, 'rb') as f:
            bm25_state = pickle.load(f)
        
        # 저장된 토크나이즈 결과 사용
        documents = bm25_state['documents']
        tokenized_corpus = bm25_state['tokenized_corpus']
        self.logger.info(f"BM25 로드 완료: {len(documents)}개 문서")
        
        # BM25Okapi 객체 생성
        bm25_index = BM25Okapi(tokenized_corpus)

        return bm25_index

    def _init_retriever(self):
        """
        하이브리드 retriever 초기화
        FAISS + BM25 ensemble → Contextual Compression
        """
        faiss_dir = self.research_storage_path / "faiss"
        bm25_path = self.research_storage_path / "bm25_index.pkl"
        lock_path = self.research_storage_path / ".init.lock"

        self.research_storage_path.mkdir(exist_ok=True)

        start = time.time()
        with open(lock_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                if faiss_dir.exists() and bm25_path.exists():
                    self._load(faiss_dir, bm25_path)
                else:
                    self._build_and_save(faiss_dir, bm25_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                self.logger.info(f"ResearchRAG 하이브리드 검색 초기화 소요 시간: {time.time() - start:.2f}초")
        self.logger.info("ResearchRAG 하이브리드 검색 로드 완료")

    def _load_research_data(self):
        """research_data 파일 로드"""
        pattern = str(self.rag_data_path / "research_data_*.jsonl")
        json_files = glob.glob(pattern)
        
        if not json_files:
            self.logger.warning(f"패턴에 맞는 파일 없음: {pattern}")
            return []
            
        all_data = []
        for file_path in json_files:
            count = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    all_data.append(json.loads(line))
                    count += 1
            self.logger.info(f"Loaded {count} items from {Path(file_path).name}")
        
        self.logger.info(f"Total loaded: {len(all_data)} items from {len(json_files)} files")
        return all_data
        
    def _build_docs(self, data):
        """크롤링 데이터를 Document 형식으로 변환"""

        bm25_docs = []
        for record in data:
            search_text = f"{record['metadata']['title']} {' '.join(record['metadata']['keywords'])}".strip()
            bm25_metadata = record["metadata"]
            bm25_metadata["page_content"] = record["page_content"]
            bm25_docs.append(Document(
                page_content=search_text,
                metadata=bm25_metadata
            ))

        dense_docs = []
        for record in data:
            dense_docs.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"]
                )
            )
        return bm25_docs, dense_docs
    
    def _create_vectorstore_in_batches(self, docs: List[Document], batch_size: int = 32):
        """
        배치 단위로 FAISS vectorstore 생성
        OpenAI embedding API 토큰 제한 회피
        """
        self.logger.info(f"총 {len(docs)}개 문서를 {batch_size}개씩 배치 처리")
        
        vectorstore = None
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.logger.info(f"배치 {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} 처리 중 ({len(batch)}개 문서)...")
            
            if vectorstore is None:
                # 첫 배치로 vectorstore 생성
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                # 기존 vectorstore에 추가
                batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                vectorstore.merge_from(batch_vectorstore)
        
        self.logger.info("모든 배치 처리 완료")
        return vectorstore
    
    def _build_and_save(self, faiss_dir: Path, bm25_path: Path):
        """하이브리드 검색 구축 및 저장"""
        data = self._load_research_data()
        bm25_docs, dense_docs = self._build_docs(data)

        # 1. BM25 state (이미 있으면 로드, 없으면 생성)
        if bm25_path.exists():
            self.logger.info("기존 BM25 문서 로드 중...")
            bm25_index = self._load_bm25_index(bm25_path)
            self.logger.info("BM25 문서 로드 완료")
        else:
            self.logger.info("BM25 인덱스 생성 중...")
            # 토크나이징 수행 및 저장
            tokenized_corpus = []
            for doc in bm25_docs:                
                tokens = self.kiwi.extract_nouns_with_ngram(doc.page_content)
                tokenized_corpus.append(tokens)
            # 상태 저장
            bm25_state = {
                'documents': bm25_docs,
                'tokenized_corpus': tokenized_corpus,
                'k': self.k,
            }
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_state, f, protocol=4)
            self.logger.info(f"BM25 인덱스 저장 완료: {bm25_path}")
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_state, f)
            self.logger.info(f"BM25 문서 저장 완료: {bm25_path}")
            bm25_index = BM25Okapi(tokenized_corpus)

        bm25_retriever = BM25Retriever(
            vectorizer=bm25_index,
            docs=bm25_docs,
            k=self.k,
            preprocess_func=self.kiwi.extract_nouns_with_ngram
        )
    
        # 2. FAISS vectorstore (이미 있으면 로드, 없으면 생성)
        if faiss_dir.exists() and (faiss_dir / "index.faiss").exists():
            self.logger.info("기존 FAISS vectorstore 로드 중...")
            vectorstore = FAISS.load_local(
                str(faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("FAISS 로드 완료")
        else:
            self.logger.info("FAISS vectorstore 생성 중 (배치 처리)...")
            vectorstore = self._create_vectorstore_in_batches(dense_docs)
            faiss_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(faiss_dir))
            self.logger.info(f"FAISS 저장 완료: {faiss_dir}")
        
        # 3. Ensemble retriever
        self.logger.info("Ensemble retriever 생성 중...")
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # 4. Contextual Compression
        self.logger.info("Contextual Compression 적용 중...")
        compressor = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.thresholds
        )
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        self.logger.info("하이브리드 검색 구축 완료")

    def _load(self, faiss_dir: Path, bm25_path: Path):
        """저장된 하이브리드 검색 로드"""
        # 1. FAISS vectorstore 로드
        self.logger.info("FAISS vectorstore 로드 중...")
        vectorstore = FAISS.load_local(
            str(faiss_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        
        # 2. bm25_retriever 로드
        self.logger.info("bm25_retriever 로드 중...")
        with open(bm25_path, 'rb') as f:
            bm25_state = pickle.load(f)
        bm25_index = BM25Okapi(bm25_state['tokenized_corpus'])
        
        bm25_retriever = BM25Retriever(
            vectorizer=bm25_index,
            docs=bm25_state['documents'],
            k=self.k,
            preprocess_func=self.kiwi.extract_nouns_with_ngram
        )
        
        # 3. Ensemble retriever
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # 4. Contextual Compression
        compressor = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.thresholds
        )
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        self.logger.info("하이브리드 검색 로드 완료")

    # -------------------------------
    # public 함수
    # -------------------------------

    async def asearch(self, query: str, top_k: int = 5) -> List[Document]:
        """하이브리드 검색 실행"""
        if not self.retriever:
            raise ValueError("Retriever가 초기화되지 않음")
        
        results = await self.retriever.ainvoke(query)
        
        results_dict = [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in results
        ]
        final_results = []
        for result in results_dict[:top_k]:
            summary = result.get("metadata", {}).get("page_content")
            if summary == None:
                summary = result.get("page_content")

            final_results.append({
                "url": result.get("metadata", {}).get("url"),
                "category": result.get("metadata", {}).get("category"),
                "title": result.get("metadata", {}).get("title"),
                "authors": result.get("metadata", {}).get("authors"),
                "summary": summary
            })
        return final_results