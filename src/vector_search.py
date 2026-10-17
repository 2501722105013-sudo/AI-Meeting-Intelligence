"""
Module 7: Meeting Similarity Search (AI Memory Engine).
Uses Sentence-BERT (all-MiniLM-L6-v2) embeddings and FAISS vector database
to index historical meetings and retrieve relevant context for natural language queries.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MeetingVectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.index = None
        self.metadata_store = []
        self.vector_dim = 384
        self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.embedding_model_name}'...")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using TF-IDF vector fallback.")
            self.embedder = None

        # Initialize FAISS Index
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.vector_dim)
        except Exception as e:
            logger.warning(f"FAISS not available ({e}). Using Numpy cosine similarity fallback.")
            self.index = None

    def add_meeting(self, meeting_data: Dict[str, Any]):
        """
        Add a processed meeting record to vector index.
        meeting_data expects:
        {
          "meeting_id": "...",
          "date": "...",
          "department": "...",
          "summary": "...",
          "transcript": "...",
          "topics": ["..."],
          "decisions": ["..."]
        }
        """
        # Formulate rich text chunk for semantic embedding
        summary_str = meeting_data.get("summary", "")
        if isinstance(summary_str, dict):
            summary_str = summary_str.get("summary_text", "")
            
        topics_raw = meeting_data.get("topics", [])
        if isinstance(topics_raw, list):
            topics_str = ", ".join([
                t.get("topic_name", str(t)) if isinstance(t, dict) else str(t)
                for t in topics_raw
            ])
        else:
            topics_str = str(topics_raw)
        decisions_str = "; ".join([d.get("decision", "") if isinstance(d, dict) else str(d) for d in meeting_data.get("decisions", [])])

        chunk_text = f"Department: {meeting_data.get('department', '')}. Topics: {topics_str}. Summary: {summary_str}. Decisions: {decisions_str}."

        if self.embedder is not None:
            embedding = self.embedder.encode([chunk_text], show_progress_bar=False)[0]
        else:
            # Simple TF-IDF / Hash embedding fallback vector
            embedding = self._simple_hash_vector(chunk_text)

        # Normalize vector for cosine distance equivalent
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding_matrix = np.array([embedding], dtype=np.float32)

        if self.index is not None:
            self.index.add(embedding_matrix)
        
        # Save metadata record along with embedding
        record = dict(meeting_data)
        record["_embedding"] = embedding.tolist()
        self.metadata_store.append(record)

    def search_meetings(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search previous meetings for natural language queries (e.g. 'What was discussed about marketing last month?')
        """
        if not self.metadata_store:
            return []

        if self.embedder is not None:
            query_emb = self.embedder.encode([query], show_progress_bar=False)[0]
        else:
            query_emb = self._simple_hash_vector(query)

        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        query_matrix = np.array([query_emb], dtype=np.float32)

        results = []

        if self.index is not None and self.index.ntotal > 0:
            k_num = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_matrix, k_num)
            
            for idx_pos, meta_idx in enumerate(indices[0]):
                if meta_idx < len(self.metadata_store):
                    rec = dict(self.metadata_store[meta_idx])
                    rec.pop("_embedding", None)
                    # Convert L2 distance to percentage similarity score
                    score = float(max(0.0, 1.0 - (distances[0][idx_pos] / 2.0)))
                    rec["similarity_score"] = round(score * 100, 2)
                    results.append(rec)
        else:
            # Numpy cosine similarity fallback
            scores = []
            for rec in self.metadata_store:
                emb = np.array(rec["_embedding"], dtype=np.float32)
                sim = float(np.dot(query_emb, emb))
                scores.append((sim, rec))
            scores.sort(key=lambda x: x[0], reverse=True)
            for sim, rec in scores[:top_k]:
                clean_rec = dict(rec)
                clean_rec.pop("_embedding", None)
                clean_rec["similarity_score"] = round(max(0.0, sim) * 100, 2)
                results.append(clean_rec)

        return results

    def _simple_hash_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % self.vector_dim
            vec[idx] += 1.0
        return vec

# Global Vector Memory Singleton
_vector_store = None

def get_vector_store() -> MeetingVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MeetingVectorStore()
    return _vector_store
