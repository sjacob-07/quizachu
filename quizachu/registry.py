from quizachu.params import *
from google.cloud import storage
from pathlib import Path

def get_generate_weights_path():

    path = Path(LOCAL_MODELS_PATH + "/" + GENERATE_MODEL_WEIGHTS_NAME)
    if path.is_file():
        return str(path)

    print(f"\nLoad latest model from GCS...")

    client = storage.Client()
    blobs = list(client.get_bucket(MODELS_BUCKET).list_blobs(prefix="generate"))
    print(blobs)

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        weights_local_path = os.path.join(LOCAL_MODELS_PATH, latest_blob.name)
        latest_blob.download_to_filename(weights_local_path)

        print("✅ Latest weights retrieved from cloud storage, path returned")

        return weights_local_path
    except:
        print(f"\n❌ No model found in GCS bucket {MODELS_BUCKET}")

        return None

def get_scoring_model_path():
    path = Path(LOCAL_MODELS_PATH + "/score_model/score_model_basic.h5")
    if path.is_file():
        return path

    print(f"\nLoad latest model from GCS...")

    client = storage.Client()
    blobs = list(client.get_bucket(MODELS_BUCKET).list_blobs(prefix="score_model/score_model_basic.h5"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        weights_local_path = os.path.join(LOCAL_MODELS_PATH, latest_blob.name)
        latest_blob.download_to_filename(weights_local_path)

        print("✅ Latest model retrieved from cloud storage, path returned")

        return weights_local_path
    except:
        print(f"\n❌ No model found in GCS bucket {MODELS_BUCKET}")

        return None
