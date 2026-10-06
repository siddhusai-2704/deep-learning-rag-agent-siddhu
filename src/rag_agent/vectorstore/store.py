"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.
        """
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB initialized: collection='{}', items={}",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            logger.error("Failed to initialize ChromaDB: {}", exc)
            raise RuntimeError(
                f"Could not initialize ChromaDB: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists.
        """
        result = self._collection.get(ids=[chunk_id])

        ids = result.get("ids", [])

        return chunk_id in ids

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.
        """
        result = IngestionResult()

        batch_size = 100

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            ids = []
            documents = []
            metadatas = []
            texts_to_embed = []

            for chunk in batch:
                try:
                    if self.check_duplicate(chunk.chunk_id):
                        result.skipped += 1
                        continue

                    ids.append(chunk.chunk_id)
                    documents.append(chunk.chunk_text)
                    texts_to_embed.append(chunk.chunk_text)

                    metadata = chunk.metadata

                    if hasattr(metadata, "to_dict"):
                        metadata_dict = metadata.to_dict()
                    else:
                        metadata_dict = metadata.__dict__.copy()

                    # Chroma metadata values must be primitive types.
                    # Convert related_topics list to a string if necessary.
                    if isinstance(
                        metadata_dict.get("related_topics"), list
                    ):
                        metadata_dict["related_topics"] = ",".join(
                            metadata_dict["related_topics"]
                        )

                    metadatas.append(metadata_dict)

                except Exception as exc:
                    error_message = (
                        f"{chunk.chunk_id}: {exc}"
                    )
                    result.errors.append(error_message)
                    logger.error(
                        "Error preparing chunk {}: {}",
                        chunk.chunk_id,
                        exc,
                    )

            if not ids:
                continue

            try:
                embeddings = self._embeddings.embed_documents(
                    texts_to_embed
                )

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                result.ingested += len(ids)

            except Exception as exc:
                error_message = (
                    f"Batch ingestion failed: {exc}"
                )
                result.errors.append(error_message)

                logger.error(
                    "Batch ingestion failed: {}",
                    exc,
                )

        logger.info(
            "Ingestion complete: ingested={}, skipped={}, errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.
        """
        k = k or self._settings.retrieval_k

        where_filter = None

        if topic_filter and difficulty_filter:
            where_filter = {
                "$and": [
                    {"topic": topic_filter},
                    {"difficulty": difficulty_filter},
                ]
            }
        elif topic_filter:
            where_filter = {"topic": topic_filter}
        elif difficulty_filter:
            where_filter = {"difficulty": difficulty_filter}

        query_embedding = self._embeddings.embed_query(query_text)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        result_ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []

        for chunk_id, document, metadata, distance in zip(
            result_ids,
            documents,
            metadatas,
            distances,
        ):
            # Chroma cosine distance:
            # similarity = 1 - distance
            score = 1.0 - distance

            if score < self._settings.similarity_threshold:
                continue

            if metadata is None:
                metadata = {}

            # Convert related_topics back to a list.
            related_topics = metadata.get("related_topics", [])

            if isinstance(related_topics, str):
                if related_topics.strip():
                    related_topics = [
                        topic.strip()
                        for topic in related_topics.split(",")
                    ]
                else:
                    related_topics = []

            metadata["related_topics"] = related_topics

            chunk_metadata = ChunkMetadata(
                topic=metadata.get("topic", ""),
                difficulty=metadata.get("difficulty", ""),
                type=metadata.get("type", ""),
                source=metadata.get("source", ""),
                related_topics=related_topics,
                is_bonus=metadata.get("is_bonus", False),
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=chunk_metadata,
                    score=score,
                )
            )

        retrieved.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return retrieved

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.
        """
        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        documents = {}

        for metadata in metadatas:
            if not metadata:
                continue

            source = metadata.get("source", "unknown")
            topic = metadata.get("topic", "")

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": topic,
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["source"].lower(),
        )

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        chunks = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            metadata = metadata or {}

            related_topics = metadata.get("related_topics", [])

            if isinstance(related_topics, str):
                if related_topics.strip():
                    related_topics = [
                        topic.strip()
                        for topic in related_topics.split(",")
                    ]
                else:
                    related_topics = []

            chunk_metadata = ChunkMetadata(
                topic=metadata.get("topic", ""),
                difficulty=metadata.get("difficulty", ""),
                type=metadata.get("type", ""),
                source=metadata.get("source", source),
                related_topics=related_topics,
                is_bonus=metadata.get("is_bonus", False),
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document or "",
                    metadata=chunk_metadata,
                )
            )

        # Chunk IDs are deterministic but do not encode document position.
        # Sorting by chunk ID provides deterministic display order.
        chunks.sort(key=lambda chunk: chunk.chunk_id)

        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.
        """
        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        topics = set()
        sources = set()
        bonus_topics_present = False

        for metadata in metadatas:
            if not metadata:
                continue

            topic = metadata.get("topic")
            source = metadata.get("source")
            is_bonus = metadata.get("is_bonus", False)

            if topic:
                topics.add(topic)

            if source:
                sources.add(source)

            if is_bonus:
                bonus_topics_present = True

        return {
            "total_chunks": self._collection.count(),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.
        """
        results = self._collection.get(
            where={"source": source}
        )

        ids = results.get("ids", [])

        if not ids:
            return 0

        self._collection.delete(ids=ids)

        logger.info(
            "Deleted {} chunks from document '{}'",
            len(ids),
            source,
        )

        return len(ids)