import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_embeddings(texts, model, tokenizer, normalize=True):
    """
    Generate embeddings for a list of texts using a pre-trained model.

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
    encoded_input = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    encoded_input = encoded_input.to(device)

    # Get the embeddings for the texts
    with torch.no_grad():
        model_output = model(**encoded_input)

        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]


    # Normalize embeddings if required
    if normalize:
        sentence_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # convert to numpy array
    sentence_embeddings = sentence_embeddings.cpu().numpy()
    ret_value = sentence_embeddings.tolist()

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
        "embeddings_tokenizer": embeddings_tokenizer
    }

    return model


def predict_fn(texts, model):
    logger.info("predict_fn")

    embeddings_model = model["embeddings_model"]
    embeddings_tokenizer = model["embeddings_tokenizer"]

    ret_value = generate_embeddings(texts, embeddings_model, embeddings_tokenizer)

    return ret_value