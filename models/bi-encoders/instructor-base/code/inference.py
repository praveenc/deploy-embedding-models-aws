from InstructorEmbedding import INSTRUCTOR
import torch


def model_fn(model_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = INSTRUCTOR(model_dir)
    model.eval()
    model.to(device)

    return model


def predict_fn(texts, model):
    ret_value = model.encode(texts)
    return ret_value