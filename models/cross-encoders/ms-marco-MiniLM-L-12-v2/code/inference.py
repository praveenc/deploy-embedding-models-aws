import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
{
    "sentence": "I love Berlin",
    "candidates": ["I love Paris", "I love London"]
}
"""


def model_fn(model_dir):
    logger.info("model_fn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    model = {
        "cross_encoder_model": model,
        "cross_encoder_tokenizer": tokenizer,
    }

    return model


def predict_fn(input_object, model):
    logger.info("predict_fn")

    cross_encoder_model = model["cross_encoder_model"]
    cross_encoder_tokenizer = model["cross_encoder_tokenizer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentence = input_object["sentence"]
    candidates = input_object["candidates"]

    data = [[sentence, candidate] for candidate in candidates]

    features = cross_encoder_tokenizer(
        data, padding=True, truncation=True, return_tensors="pt"
    )
    features = features.to(device)

    with torch.no_grad():
        scores = cross_encoder_model(**features).logits.cpu().numpy()
        ret_value = scores.tolist()

    return ret_value