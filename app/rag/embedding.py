import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

class Qwen3Embeddings(Embeddings):
    def __init__(self, CUDA_VISIBLE_DEVICES: bool):
        model_name: str = "Qwen/Qwen3-Embedding-0.6B"

        if CUDA_VISIBLE_DEVICES:
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={
                    # "attn_implementation": "flash_attention_2", 
                    "device_map": "cuda", 
                    "dtype": torch.float32, 
                    },
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "device_map": "cpu", 
                    "dtype": torch.float32, 
                    },
                tokenizer_kwargs={"padding_side": "left"},
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 임베딩 - prompt 없이
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=4)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 임베딩 - query prompt 적용
        """
        embedding = self.model.encode([text], prompt_name="query", normalize_embeddings=True)
        return embedding[0].tolist()