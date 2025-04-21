"""RAG layer for Food & Weather Buddy (v6)
– Corrige bug "MissingIndex: index with index_name 'vector' not initialized".
  Ahora intentamos leer el índice; si no existe lo creamos.
– Maneja vectores como listas, bytes o strings.
"""
from __future__ import annotations

import ast
import os
from functools import lru_cache
from typing import List, Tuple
import re
import numpy as np
from datasets import load_dataset, Dataset, features, load_from_disk
from datasets.search import MissingIndex
from sentence_transformers import SentenceTransformer

DATASET_ID = "somosnlp/RecetasDeLaAbuela"
DATASET_VERSION = 'version_1'
FILE_ID = os.getenv("RECETAS_CACHE", "recetas_cached")
TEXT_MODEL = os.getenv("RECETAS_MODEL", "sentence-transformers/all-mpnet-base-v2")
SAMPLE_SIZE = int(os.getenv("FOODBUDDY_SAMPLE", "10000"))
BATCH_SIZE = int(os.getenv("RECETAS_BATCH", "1000"))

_NUM_DTYPES = {f"float{b}" for b in (16, 32, 64)} | {f"int{b}" for b in (8, 16, 32, 64)} | {
    f"uint{b}" for b in (8, 16, 32, 64)
}

# ---------- helpers ---------------------------------------------------------

_float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")



def _is_numeric_sequence(feat: features.Feature) -> bool:
    if isinstance(feat, features.Sequence):
        return _is_numeric_sequence(feat.feature)
    return isinstance(feat, features.Value) and feat.dtype in _NUM_DTYPES


# ---------- loaders ---------------------------------------------------------

@lru_cache(maxsize=1)
def _load_dataset() -> Dataset:
    if os.path.isdir(FILE_ID):
        ds = load_from_disk(FILE_ID)
        print("Loaded file {FILE_ID}")
    else:
        try:
            ds = load_dataset(DATASET_ID, DATASET_VERSION, split="train")
        except Exception:
            print("[FoodRAG] Full download failed → streaming sample.")
            stream = load_dataset(DATASET_ID, DATASET_VERSION, split="train", streaming=True)
            ds = Dataset.from_generator(lambda: (x for i, x in enumerate(stream) if i < SAMPLE_SIZE))

        text_cols = [c for c, feat in ds.features.items() if feat.dtype == "string"]
        print("[RAG] Text columns:", text_cols)

        def _join(batch: Dict[str, List[Any]]):
            batch["joined_text"] = [" \n ".join(str(batch[col][i]) for col in text_cols if batch[col][i])
                                    for i in range(len(batch[text_cols[0]]))]
            return batch

        ds = ds.map(_join, batched=True, batch_size=BATCH_SIZE, desc="Concatenating text")

        # embeddings
        encoder = SentenceTransformer(TEXT_MODEL)

        def _embed(batch):
            emb = encoder.encode(batch["joined_text"], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            batch["vector"] = [e.astype(np.float32) for e in emb]
            return batch

        ds = ds.map(_embed, batched=True, batch_size=BATCH_SIZE, desc="Embedding", num_proc=6)

        ds.set_format(type="numpy", columns=["vector"], output_all_columns=True)
        ds.save_to_disk(FILE_ID)
        # Ensure Faiss index exists
    try:
        ds.get_index("vector")
    except MissingIndex:
        ds.add_faiss_index(column="vector")


        
    return ds

@lru_cache(maxsize=1)
def _load_encoder() -> SentenceTransformer:
    model = SentenceTransformer(TEXT_MODEL)
    return model

# ---------- main class ------------------------------------------------------

class FoodRAG:
    def __init__(self):
        self.ds = _load_dataset()
        self.model = _load_encoder()

    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).astype("float32")

    def search(self, query: str, limit: int = 5) -> List[Tuple[float, dict]]:
        q_emb = self._encode(query)
        scores, samples = self.ds.get_nearest_examples("vector", q_emb, k=limit)
        rows = [{col: samples[col][i] for col in samples} for i in range(len(scores))]
        return list(zip(scores, rows))

