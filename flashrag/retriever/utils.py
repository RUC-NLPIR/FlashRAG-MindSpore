import json
from mindnlp.transformers import AutoModel, AutoTokenizer
from mindspore.ops._primitive_cache import _get_cache_prim
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindnlp.dataset import load_dataset
from typing import Dict, Any, Union, List, Dict
import numpy as np

def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert numpy objects in nested dictionaries or lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to native Python scalars
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj  # Return the object as-is if it's neither a dict, list, nor numpy type


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


def set_default_instruction(model_name, is_query=True, is_zh=False):
    instruction = ""
    if "e5" in model_name.lower():
        if is_query:
            instruction = "query: "
        else:
            instruction = "passage: "

    if "bge" in model_name.lower():
        if is_query:
            if "zh" in model_name.lower() or is_zh:
                instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                instruction = "Represent this sentence for searching relevant passages: "

    return instruction


def parse_query(model_name, query_list, instruction=None):
    """
    processing query for different encoders
    """

    def is_zh(str):
        import unicodedata

        zh_char = 0
        for c in str:
            try:
                if "CJK" in unicodedata.name(c):
                    zh_char += 1
            except:
                continue
        if len(str) == 0:
            return False
        if zh_char / len(str) > 0.2:
            return True
        else:
            return False

    if isinstance(query_list, str):
        query_list = [query_list]

    if instruction is not None:
        instruction = instruction.strip() + " "
    else:
        instruction = set_default_instruction(model_name, is_query=True, is_zh=is_zh(query_list[0]))
    print(f"Use `{instruction}` as retreival instruction")

    query_list = [instruction + query for query in query_list]

    return query_list


def load_corpus(corpus_path: str):
    corpus = load_dataset(
            'json',
            data_files=corpus_path,
            split="train")
    return corpus.source


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    docs = []
    for idx in doc_idxs:
        doc_item = corpus[int(idx)]
        if len(doc_item) == 2:
            doc_item = {"id": doc_item[0], "contents": doc_item[1]}
        elif len(doc_item) == 3:
            doc_item = {"id": doc_item[0], "title": doc_item[1], "contents": doc_item[2]}
        else:
            assert False
        docs.append(doc_item)
    return docs

def parse_image(image):
    from PIL import Image

    if isinstance(image, str):
        if image.startswith("http"):
            import requests

            image = Image.open(requests.get(image, stream=True).raw)
        else:
            image = Image.open(image)
    return image
