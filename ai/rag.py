import os
from .embeddings import get_model, embed_texts, build_faiss_index, search_index
from .embeddings import build_index_from_texts, save_faiss_index, load_faiss_index
from .chunker import chunk_text_by_sentences


class RAGStore:
    def __init__(self, texts=None, index_path_prefix: str = None, model_name: str = None):
        """If index_path_prefix exists on disk, load persisted index and metadata.
        Otherwise, build index from provided texts (will chunk) and keep in memory.
        """
        self.model = get_model(model_name) if model_name else get_model()
        self.index = None
        self.metadata = []
        if index_path_prefix and os.path.exists(index_path_prefix + '.index'):
            self.index, self.metadata = load_faiss_index(index_path_prefix)
        elif texts:
            # build chunks
            chunks = []
            for t in texts:
                c = chunk_text_by_sentences(t)
                for seg in c:
                    chunks.append(seg['text'])
            self.index, self.embs, self.metadata = build_index_from_texts(chunks, model=self.model)
        else:
            raise ValueError('Either texts or index_path_prefix must be provided')

    def persist(self, path_prefix: str):
        if self.index is None:
            raise RuntimeError('No index to save')
        save_faiss_index(self.index, self.metadata, path_prefix)

    def query(self, q, k=4):
        q_emb = self.model.encode([q], convert_to_numpy=True)[0]
        D, I = search_index(self.index, q_emb, k=k)
        results = []
        for score, idx in zip(D, I):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append((meta.get('text', ''), float(score)))
        return results
