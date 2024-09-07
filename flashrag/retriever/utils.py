import json
from mindnlp.transformers import AutoModel, AutoTokenizer
from mindspore.ops._primitive_cache import _get_cache_prim
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindnlp.dataset import load_dataset


def load_model(
        model_path: str,
        use_fp16: bool = False
    ):
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        if attention_mask is not None:
            mask = ops.ExpandDims()(attention_mask, -1).astype(ms.bool_)
            last_hidden = mnp.where(mask, last_hidden_state, mnp.zeros_like(last_hidden_state))
            return last_hidden.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
        else:
            return last_hidden_state.mean(axis=1)
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_corpus(corpus_path: str):
    corpus = load_dataset(
            'json',
            data_files=corpus_path,
            split="train")
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    doc_dict = {}
    idx_set = set(doc_idxs)

    for i, doc in enumerate(corpus.create_dict_iterator()):
        if i in idx_set:
            doc_str = {key: str(value) for key, value in doc.items()}
            doc_dict[i] = doc_str
            idx_set.remove(i)
        if not idx_set:
            break

    docs = [doc_dict[idx] for idx in doc_idxs if idx in doc_dict]
    return docs
