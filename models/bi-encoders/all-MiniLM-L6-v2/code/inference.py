import torch
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
["I love Berlin", "I love Paris", "I love London"]
"""


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def model_fn(model_dir):
    logger.info("model_fn")
    print(f"Model dir: {model_dir}")
    sorted_files = sorted(Path(model_dir).rglob("*.*"))
    print(f"Sorted files: {sorted_files}")
    logger.info(f"Sorted files: {sorted_files}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    model_obj = {
        "embeddings_model": model,
        "embeddings_tokenizer": tokenizer,
    }

    return model_obj


def predict_fn(texts, model):
    logger.info("predict_fn")

    embeddings_model = model["embeddings_model"]
    embeddings_tokenizer = model["embeddings_tokenizer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoded_input = embeddings_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded_input = encoded_input.to(device)

    with torch.no_grad():
        model_output = embeddings_model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    response = sentence_embeddings.cpu().numpy()
    ret_value = response.tolist()

    return ret_value
