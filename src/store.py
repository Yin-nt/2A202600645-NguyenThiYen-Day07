from __future__ import annotations

import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            client = chromadb.Client()
            
            # Xoá collection cũ (nếu có) để tránh rò rỉ dữ liệu giữa các bài test
            try:
                client.delete_collection(name=self._collection_name)
            except Exception:
                pass
                
            self._collection = client.create_collection(name=self._collection_name)
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
            embedding = self._embedding_fn(doc.content)
            m = doc.metadata.copy() if doc.metadata else {}
            m["doc_id"] = doc.id # Giấu ID gốc vào metadata
            return {
                "id": str(uuid.uuid4()), # Sinh ID ngẫu nhiên cho chunk để không bao giờ bị đè
                "content": doc.content,
                "embedding": embedding,
                "metadata": m
            }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embed = self._embedding_fn(query)
        scored_records = []
        
        for record in records:
            score = _dot(query_embed, record["embedding"])
            scored_records.append({"score": score, "record": record})
            
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in scored_records[:top_k]:
            res = item["record"].copy()
            res["score"] = item["score"] 
            results.append(res)
            
        return results

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return

        if self._use_chroma:
            ids = [str(uuid.uuid4()) for _ in docs] # Sinh UUID ngẫu nhiên
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            
            metadatas = []
            for doc in docs:
                m = doc.metadata.copy() if doc.metadata else {}
                m["doc_id"] = doc.id # Giấu ID gốc vào metadata
                metadatas.append(m)
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma:
            query_embed = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embed],
                n_results=top_k
            )
            
            parsed_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    raw_distance = results['distances'][0][i] if results.get('distances') else 0.0
                    # Chuyển đổi distance (càng nhỏ càng tốt) thành score (càng lớn càng tốt)
                    score = 1.0 / (1.0 + raw_distance)
                    
                    parsed_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0][i] else {},
                        "score": score
                    })
            return parsed_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            query_embed = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embed],
                n_results=top_k,
                where=metadata_filter
            )
            
            parsed_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    raw_distance = results['distances'][0][i] if results.get('distances') else 0.0
                    score = 1.0 / (1.0 + raw_distance)
                    
                    parsed_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0][i] else {},
                        "score": score
                    })
            return parsed_results
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
                    
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma:
            initial_count = self._collection.count()
            # Xoá dựa vào doc_id nằm trong metadata
            self._collection.delete(where={"doc_id": doc_id}) 
            return self._collection.count() < initial_count
        else:
            initial_count = len(self._store)
            self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count