import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def average_pooling(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Calculate the average pooling of the last hidden states.

    Args:
        last_hidden_states (Tensor): Tensor containing the last hidden states.
        attention_mask (Tensor): Tensor containing the attention mask.

    Returns:
        Tensor: Tensor containing the average pooled embeddings.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def calculate_scores(query_embeddings, corpus_embeddings, queries, corpus):
    """
    Calculate the cosine similarity scores between queries and corpus sentences.

    Args:
        query_embeddings (Tensor): Tensor containing the embeddings of query sentences.
        corpus_embeddings (Tensor): Tensor containing the embeddings of corpus sentences.
        queries (List[str]): List of query sentences.
        corpus (List[str]): List of corpus sentences.

    Returns:
        List[Tuple[str, str, float]]: List of tuples containing query, hit, and similarity score.
    """
    # Calculate the similarity scores
    # scores = torch.mm(query_embeddings, corpus_embeddings.T) * 100
    scores = torch.mm(query_embeddings, corpus_embeddings.T)

    # Get the top 3 similarity scores for each query
    results = []
    for i, query in enumerate(queries):
        _, indices = torch.topk(scores[i], 3)
        for idx in indices:
            results.append((query, corpus[idx], scores[i][idx].item()))

    return results


def calculate_embeddings(texts, model, tokenizer, normalize=True):
    """
    Calculate the embeddings for a list of texts using a pre-trained model.

    Args:
        texts (List[str]): List of texts to calculate embeddings for.
        model (AutoModel): Pre-trained model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.
        normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.

    Returns:
        Tensor: Tensor containing the embeddings for the texts.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize the texts
    text_dict = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    text_dict = text_dict.to(device)

    # Get the embeddings for the texts
    with torch.no_grad():
        text_outputs = model(**text_dict)

    text_embeddings = average_pooling(
        text_outputs.last_hidden_state, text_dict["attention_mask"]
    )

    # Normalize embeddings if required
    if normalize:
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # convert to numpy array and then to list
    text_embeddings = text_embeddings.cpu().numpy()
    ret_value = text_embeddings.tolist()

    return ret_value


def model_fn(model_dir):
    logger.info("model_fn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    embeddings_model = AutoModel.from_pretrained(model_dir)
    embeddings_model.eval()
    embeddings_model.to(device)

    model = {
        "embeddings_model": embeddings_model,
        "embeddings_tokenizer": embeddings_tokenizer,
    }

    return model


def predict_fn(texts, model):
    logger.info("predict_fn")

    embeddings_model = model["embeddings_model"]
    embeddings_tokenizer = model["embeddings_tokenizer"]

    ret_value = calculate_embeddings(texts, embeddings_model, embeddings_tokenizer)

    return ret_value
