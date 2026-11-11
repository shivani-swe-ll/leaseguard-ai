import numpy as np
import pytest

from leaseguard.retrieval.bm25 import BM25, stem, tokenize
from leaseguard.retrieval.hybrid import MODES, HybridRetriever, expand_query
from leaseguard.retrieval.vectors import APIEmbedder, LocalEmbedder, VectorIndex


def test_stem_and_tokenize():
    assert stem("deposits") == "deposit"
    assert stem("subletting") == "sublet"
    assert "what" not in tokenize("What is a damage deposit?")


def test_bm25_prefers_matching_doc():
    bm = BM25(["damage deposit rules", "rent increase guideline", "pets and animals"])
    scores = bm.scores("is a damage deposit legal")
    assert scores.index(max(scores)) == 0


def test_local_embedder_handles_word_forms():
    emb = LocalEmbedder().fit(["sublet assign", "rent increase"])
    a, b, c = emb.embed(["subletting", "sublet", "rent increase"])
    assert float(a @ b) > float(a @ c)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_vector_index_save_and_load(tmp_path):
    emb = LocalEmbedder()
    idx = VectorIndex.build(["x", "y"], ["damage deposit", "late fee"], emb)
    idx.save(tmp_path / "i.npz")
    loaded = VectorIndex.load(tmp_path / "i.npz")
    assert loaded.ids == ["x", "y"] and loaded.embedder_name == "local-char-ngram"
    assert loaded.search(emb.embed(["deposit for damage"])[0], k=1)[0][0] == "x"


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_find_the_damage_deposit_rule(retriever, mode):
    assert retriever.search("Can my landlord ask for a damage deposit?", k=3, mode=mode)[0].rule_id == "R05"


def test_vector_search_helps_with_word_forms(retriever):
    # keyword search alone misses "sub-let"; vector search finds the sublet rules
    vec_ids = [h.rule_id for h in retriever.search("The tenant shall not sub-let the flat", k=3, mode="vector")]
    assert "R19" in vec_ids


def test_rerank_marks_trigger_words(retriever):
    hit = retriever.search("A late fee of $40 applies after the 5th.", k=1)[0]
    assert hit.rule_id == "R08" and hit.trigger


def test_query_expansion():
    assert "enter" in expand_query("my landlord comes in whenever")


def test_bad_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_api_embedder_against_fake_server(fake_server):
    url, state = fake_server
    emb = APIEmbedder("embed-model", url, "key")
    vecs = emb.embed(["a", "b"])
    assert vecs.shape == (2, 3)
    assert state["requests"][0]["path"] == "/embeddings"
    assert state["requests"][0]["body"]["model"] == "embed-model"


def test_retriever_with_custom_embedder(rules):
    class Constant:
        name = "const"

        def fit(self, corpus):
            return self

        def embed(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    r = HybridRetriever(rules, embedder=Constant())
    assert r.search("damage deposit", k=1, mode="hybrid_rerank")[0].rule_id == "R05"
