# Deploy Text Embedding Models to Amazon SageMaker

Deploy open-source text embedding models on HuggingFace to Amazon SageMaker

## Bi-Encoders

### BAAI/bge Embedding Models

>**Model Card:** <https://huggingface.co/BAAI/bge-large-en-v1.5>

- [BAAI/bge-base-en-v1.5](./deploy_bgebasev15_en_embedding_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/bge-base-en-v15/code/inference.py)
- [BAAI/bge-large-en-v1.5](./deploy_bgelargev15_en_embedding_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/bge-large-en-v15/code/inference.py)

### thenlper/gte Models

>**Model Card:** <https://huggingface.co/thenlper/gte-base>

- [thenlper/gte-large](./deploy_gte_large_embedding_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/gte-large/code/inference.py)
- [thenlper/gte-base](./deploy_gte_base_embedding_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/gte-base/code/inference.py)

### hkunlp/instructor Models

>**Model Card:** <https://huggingface.co/hkunlp/instructor-base>

- [hkunlp/instructor-base](./deploy_instructor_embedding_models.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/instructor-base/code/inference.py)

### sentence-transformers Models
- [sentence-transformers/all-MiniLM-L6-v2](./deploy_all_MiniLM_L6v2_embedding_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/bi-encoders/all-MiniLM-L6-v2/code/inference.py)

---


## Cross-Encoders (Re-ranking models)

### BAAI/bge Re-ranking Models

>**Model Card:** <https://huggingface.co/BAAI/bge-reranker-base>

- TBD

### sentence-transformers/cross-encoder Models

- [cross-encoder/ms-marco-MiniLM-L-12-v2](./deploy_cross_encoder_reranking_model.ipynb) notebook.
  - Inference script for this model is at [inference.py](./models/cross-encoders/ms-marco-MiniLM-L-12-v2/code/inference.py)
