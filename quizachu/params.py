import os

MODELS_BUCKET = os.environ.get("MODELS_BUCKET")

LOCAL_MODELS_PATH = os.environ.get("LOCAL_MODELS_PATH")
GENERATE_MODEL_WEIGHTS_NAME = "generate-production.h5"
