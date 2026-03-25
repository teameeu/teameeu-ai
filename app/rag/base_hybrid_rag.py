import glob
import time
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from rank_bm25 import BM25Okapi


class BaseHybridRAG(ABC):
    """
    FAISS + BM25 하이브리드 검색 기반 클래스.
    서브클래스는 data_filename과 index_subdir만 오버라이드하면 된다.
    """

    def __init__(self, app):
        self.logger = app.state.logger
        self.rag_data_path = Path(app.state.config.rag_data_path)
        self.kiwi = app.state.kiwi
        self.thresholds = 0.4
        self.k = 10
        self.embeddings = app.state.embeddings
        self.retriever: ContextualCompressionRetriever | None = None
        self.rerank_candidates = self.k * 2
        self.reranker = app.state.reranker

        # 인덱스 저장 경로: job_storage_path / index_subdir
        base = Path(app.state.config.storage_path)
        self.storage_path = base / self.index_subdir
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._init_retriever()

    # ------------------------------------------------------------------
    # 서브클래스 필수 구현
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def data_filename(self) -> str:
        """로드할 JSONL 파일명 (예: 'careernet_jobs.jsonl')"""

    @property
    @abstractmethod
    def index_subdir(self) -> str:
        """storage_path 하위 인덱스 디렉토리명 (예: 'careernet')"""

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _init_retriever(self):
        faiss_dir = self.storage_path / "faiss"
        bm25_path = self.storage_path / "bm25_index.pkl"

        start = time.time()

        if faiss_dir.exists() and bm25_path.exists():
            self._load(faiss_dir, bm25_path)
        else:
            self._build_and_save(faiss_dir, bm25_path)

        self.logger.info(f"초기화 소요 시간: {time.time() - start:.2f}초")

    def _load_research_data(self) -> list:
        pattern = str(self.rag_data_path / self.data_filename)
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

        self.logger.info(
            f"Total loaded: {len(all_data)} items from {len(json_files)} files"
        )
        return all_data

    def _build_docs(self, data):
        bm25_docs, dense_docs = [], []
        for record in data:
            metadata = {**record["metadata"], "page_content": record["page_content"]}
            bm25_docs.append(
                Document(page_content=record["page_content"], metadata=metadata)
            )
            dense_docs.append(
                Document(
                    page_content=record["page_content"], metadata=record["metadata"]
                )
            )
        return bm25_docs, dense_docs

    def _create_vectorstore_in_batches(
        self, docs: List[Document], batch_size: int = 32
    ) -> FAISS:
        """토큰 제한 회피를 위해 배치 단위로 FAISS vectorstore 생성"""
        self.logger.info(f"총 {len(docs)}개 문서를 {batch_size}개씩 배치 처리")
        vectorstore = None

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.logger.info(
                f"배치 {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} "
                f"처리 중 ({len(batch)}개 문서)..."
            )
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.merge_from(
                    FAISS.from_documents(batch, self.embeddings)
                )

        self.logger.info("모든 배치 처리 완료")
        return vectorstore

    def _build_bm25(self, bm25_docs: List[Document], bm25_path: Path) -> BM25Okapi:
        self.logger.info("BM25 인덱스 생성 중...")
        tokenized_corpus = [
            self.kiwi.extract_nouns_with_ngram(doc.page_content)
            for doc in bm25_docs
        ]
        bm25_state = {
            "documents": bm25_docs,
            "tokenized_corpus": tokenized_corpus,
            "k": self.k,
        }
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_state, f, protocol=4)
        self.logger.info(f"BM25 인덱스 저장 완료: {bm25_path}")
        return BM25Okapi(tokenized_corpus)

    def _load_bm25_index(self, bm25_path: Path) -> tuple[BM25Okapi, list]:
        self.logger.info("기존 BM25 인덱스 로드 중...")
        with open(bm25_path, "rb") as f:
            bm25_state = pickle.load(f)
        self.logger.info(f"BM25 로드 완료: {len(bm25_state['documents'])}개 문서")
        return BM25Okapi(bm25_state["tokenized_corpus"]), bm25_state["documents"]

    def _make_bm25_retriever(
        self, bm25_index: BM25Okapi, docs: list
    ) -> BM25Retriever:
        return BM25Retriever(
            vectorizer=bm25_index,
            docs=docs,
            k=self.k,
            preprocess_func=self.kiwi.extract_nouns_with_ngram,
        )

    def _make_ensemble_retriever(
        self, vectorstore: FAISS, bm25_retriever: BM25Retriever
    ) -> EnsembleRetriever:
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        return EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

    def _make_compression_retriever(
        self, ensemble_retriever: EnsembleRetriever
    ) -> ContextualCompressionRetriever:
        compressor = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.thresholds,
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )

    def _build_and_save(self, faiss_dir: Path, bm25_path: Path):
        data = self._load_research_data()
        bm25_docs, dense_docs = self._build_docs(data)

        # BM25
        if bm25_path.exists():
            bm25_index, bm25_docs = self._load_bm25_index(bm25_path)
        else:
            bm25_index = self._build_bm25(bm25_docs, bm25_path)

        # FAISS
        if faiss_dir.exists() and (faiss_dir / "index.faiss").exists():
            self.logger.info("기존 FAISS vectorstore 로드 중...")
            vectorstore = FAISS.load_local(
                str(faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.logger.info("FAISS vectorstore 생성 중 (배치 처리)...")
            vectorstore = self._create_vectorstore_in_batches(dense_docs)
            faiss_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(faiss_dir))
            self.logger.info(f"FAISS 저장 완료: {faiss_dir}")

        self.logger.info("Ensemble + Compression retriever 구성 중...")
        ensemble = self._make_ensemble_retriever(
            vectorstore, self._make_bm25_retriever(bm25_index, bm25_docs)
        )
        self.retriever = self._make_compression_retriever(ensemble)
        self.logger.info("하이브리드 검색 구축 완료")

    def _load(self, faiss_dir: Path, bm25_path: Path):
        self.logger.info("FAISS vectorstore 로드 중...")
        vectorstore = FAISS.load_local(
            str(faiss_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        bm25_index, bm25_docs = self._load_bm25_index(bm25_path)

        ensemble = self._make_ensemble_retriever(
            vectorstore, self._make_bm25_retriever(bm25_index, bm25_docs)
        )
        self.retriever = self._make_compression_retriever(ensemble)
        self.logger.info("하이브리드 검색 로드 완료")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def asearch(self, query: str, top_k: int = 5) -> List[str]:
        if not self.retriever:
            raise ValueError("Retriever가 초기화되지 않음")

        results = await self.retriever.ainvoke(query)
        candidates = results[: self.rerank_candidates]

        if not candidates:
            return []
        # reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            scores = [scores]
        
        reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc.page_content for _, doc in reranked[:top_k]]
