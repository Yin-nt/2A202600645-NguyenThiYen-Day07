from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Tách câu dựa trên các dấu câu kết thúc câu, giữ lại nghĩa bằng cách không làm mất dấu
        # Regex này chia cắt tại các vị trí có dấu chấm/hỏi/than theo sau bởi khoảng trắng hoặc xuống dòng
        sentences_raw = re.split(r'(?<=[.!?])(?:\s+|\n)', text)
        sentences = [s.strip() for s in sentences_raw if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base cases
        if len(current_text) <= self.chunk_size:
            return [current_text]
            
        if not remaining_separators:
            # Fallback nếu hết separator mà vẫn quá dài (dùng character split)
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Chia text với separator hiện tại
        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        chunks: list[str] = []
        current_chunk = ""

        for split in splits:
            # Nếu bản thân 1 mảnh đã lớn hơn chunk_size, gọi đệ quy để cắt nhỏ nó bằng các separator tiếp theo
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self._split(split, next_separators))
            else:
                # Ghép nối các mảnh nhỏ lại cho đến khi đạt giới hạn chunk_size
                sep_len = len(separator) if current_chunk else 0
                if len(current_chunk) + sep_len + len(split) <= self.chunk_size:
                    current_chunk += (separator if current_chunk else "") + split
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(a, b) = (a · b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude (to avoid division by zero).
    """
    dot_product = _dot(a, b)
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    # Guard against zero-magnitude vectors
    if magnitude_a < 1e-10 or magnitude_b < 1e-10:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """
    Compare different chunking strategies on the same text.

    Provides statistics (count, avg_length) for each strategy.
    """

    def compare(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        max_sentences: int = 3,
    ) -> dict[str, dict[str, any]]:
        """
        Run all three chunking strategies and return comparative statistics.

        Returns:
            Dictionary with keys: 'fixed_size', 'by_sentences', 'recursive'
            Each value is a dict with 'count', 'avg_length', and 'chunks' keys.
        """
        # Fixed-size chunking
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        fixed_chunks = fixed_chunker.chunk(text)

        # Sentence-based chunking
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
        sentence_chunks = sentence_chunker.chunk(text)

        # Recursive chunking
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        recursive_chunks = recursive_chunker.chunk(text)

        def _stats(chunks: list[str]) -> dict[str, any]:
            if not chunks:
                return {"count": 0, "avg_length": 0.0, "chunks": []}
            total_length = sum(len(c) for c in chunks)
            return {
                "count": len(chunks),
                "avg_length": total_length / len(chunks),
                "chunks": chunks,
            }

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
