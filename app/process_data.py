from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import json
import numpy as np


class ProcessData:

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    def __init__(
        self, processed_dir=processed_dir, raw_dir=raw_dir, embedding_model=None, database=None
    ):
        self.database = database
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else SentenceTransformer("intfloat/multilingual-e5-base")
        )

    def get_pdf_text(self, file_path):
        file_path = Path(file_path)
        document = []
        files = sorted(
            path
            for path in file_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        )

        if not files:
            raise ValueError(f"No PDF or TXT files found in {file_path}")

        for path in files:
            if path.is_file() and path.suffix.lower() == ".pdf":
                pdfReader = PdfReader(path)
                for page_number, page in enumerate(pdfReader.pages):
                    text = page.extract_text() or ""
                    document.append(
                        {
                            "file_name": path.name,
                            "page_number": page_number + 1,
                            "text": text,
                        }
                    )

            if path.is_file() and path.suffix.lower() == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    document.append(
                        {"file_name": path.name, "page_number": 1, "text": text}
                    )

        return document

    def do_chunking(self, documents):
        chunks = []

        for doc in documents:
            text = (doc["text"] or "").strip()

            if not text:
                continue

            chunks.append({
                "filename": doc["file_name"],
                "chunk_id": (
                    f"{doc['file_name']}::page-{doc['page_number']}::chunk-1"
                ),
                "page": doc["page_number"],
                "text": text,
            })

        return chunks

    def do_embedding(self, chunks, model):
        if not chunks:
            raise ValueError("No chunks to embed.")

        passages = [f"passage: {chunk['text']}" for chunk in chunks]

        token_lengths = [
            len(model.tokenizer.encode(text, truncation=False)) for text in passages
        ]
        if max(token_lengths) > model.max_seq_length:
            raise ValueError(
                "Some chunks exceed the model token limit. Reduce chunk size."
            )

        embeddings = model.encode(
            passages,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return embeddings

    def save_chunks_embeddings(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        document = self.get_pdf_text(self.raw_dir)
        chunks = self.do_chunking(document)
        embeddings = self.do_embedding(chunks, self.embedding_model)

        # Keep vectors separate from the JSON metadata.
        saved_chunks = [
            {key: value for key, value in chunk.items() if key != "embedding"}
            for chunk in chunks
        ]

        if len(saved_chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")

        (output_dir / "chunks.json").write_text(
            json.dumps(saved_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        np.save(output_dir / "embeddings.npy", embeddings)

        config = {
            "embedding_model": "intfloat/multilingual-e5-base",
            "normalized": True,
            "chunk_count": len(saved_chunks),
            "embedding_dimension": int(embeddings.shape[1]),
        }

        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        print(f"Extracted {len(document)} records and created {len(chunks)} chunks.")
        print("Embedding shape:", embeddings.shape)
        print("Saved to:", output_dir)
        if self.database is not None:
            self.database.import_saved_vectors(output_dir)
            print("Also saved to pgvector.")
        return chunks, embeddings, config


if __name__ == "__main__":
    processor = ProcessData()
    processor.save_chunks_embeddings(processor.processed_dir)
