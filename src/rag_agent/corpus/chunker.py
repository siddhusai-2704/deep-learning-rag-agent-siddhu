"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        elif suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Supported types are .pdf, .md, and .markdown."
            )

        base_metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        document_chunks = []

        for raw_chunk in raw_chunks:
            text = raw_chunk.get("text", "").strip()

            if not text:
                continue

            chunk_id = VectorStoreManager.generate_chunk_id(
                str(file_path.name),
                text,
            )

            # Create a separate metadata object for every chunk.
            metadata = ChunkMetadata(
                topic=base_metadata.topic,
                difficulty=base_metadata.difficulty,
                type=base_metadata.type,
                source=base_metadata.source,
                related_topics=list(base_metadata.related_topics),
                is_bonus=base_metadata.is_bonus,
            )

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunked '{}' into {} chunks",
            file_path.name,
            len(document_chunks),
        )

        return document_chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.
        """
        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path,
                    metadata_overrides=metadata_overrides,
                )

                all_chunks.extend(chunks)

            except Exception as exc:
                logger.error(
                    "Failed to chunk '{}': {}",
                    file_path,
                    exc,
                )

        logger.info(
            "Processed {} files into {} total chunks",
            len(file_paths),
            len(all_chunks),
        )

        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        split_documents = splitter.split_documents(documents)

        chunks = []

        for document in split_documents:
            text = document.page_content.strip()

            if not text:
                continue

            page = document.metadata.get("page", 0)

            chunks.append(
                {
                    "text": text,
                    "page": page,
                }
            )

        return chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.
        """
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = file_path.read_text(
            encoding="utf-8"
        )

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(text)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = []

        for section in sections:
            section_documents = recursive_splitter.split_documents(
                [section]
            )

            for document in section_documents:
                content = document.page_content.strip()

                if not content:
                    continue

                headers = []

                for key in ("h1", "h2", "h3"):
                    value = document.metadata.get(key)

                    if value:
                        headers.append(str(value))

                header = " > ".join(headers)

                chunks.append(
                    {
                        "text": content,
                        "header": header,
                    }
                )

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.
        """
        stem = file_path.stem

        parts = stem.rsplit("_", 1)

        if len(parts) == 2:
            topic = parts[0]
            difficulty = parts[1].lower()
        else:
            topic = stem
            difficulty = "intermediate"

        # Convert common filename formatting to readable topic names.
        topic = topic.replace("-", " ").replace("_", " ")

        bonus_topics = {
            "som",
            "boltzmannmachine",
            "boltzmann machine",
            "gan",
            "gans",
        }

        is_bonus = topic.lower() in bonus_topics

        metadata_values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": file_path.suffix.lower().lstrip("."),
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": is_bonus,
        }

        if overrides:
            for key, value in overrides.items():
                if key in metadata_values:
                    metadata_values[key] = value

        # Ensure related_topics is always a list.
        if metadata_values["related_topics"] is None:
            metadata_values["related_topics"] = []

        return ChunkMetadata(
            topic=metadata_values["topic"],
            difficulty=metadata_values["difficulty"],
            type=metadata_values["type"],
            source=metadata_values["source"],
            related_topics=metadata_values["related_topics"],
            is_bonus=metadata_values["is_bonus"],
        )