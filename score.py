# score.py (Batch Endpoint) — możliwie 1:1 jak oryginał
# Zmiany ograniczone do minimum:
# - run(): batch contract (mini_batch = lista ścieżek do plików), root=document, NUM_PREDS z env
# - obsługa błędów per plik + stały output (zawsze JSON + metadata)
# - wymuszenie JSON-serializable (żeby w output nie było pythonowego repr z apostrofami)

import os
import time
import json
import pandas as pd
from models.logistic_regression import LogisticRegression  # noqa
from models.transformer import Transformer  # noqa
import pre_processing  # noqa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

relevance_model = None
cd_logreg_model = None
cd_transformer_model = None


def init():
    global relevance_model
    global cd_logreg_model
    global cd_transformer_model

    logger.info("Initializing model...")
    model = os.getenv("AZUREML_MODEL_DIR")
    print(f"line 23: {os.listdir(model)}")
    print(f"line 24: {model}")

    model = os.path.join(model, "albot")
    print(f"line 27: {os.listdir(model)}")

    relevance_model = LogisticRegression.load(
        os.path.join(model, "logreg_relevance.joblib")
    )

    cd_logreg_model = LogisticRegression.load(
        os.path.join(model, "logreg_cd.joblib")
    )

    transformer = os.path.join(model, "transformer_model")

    model_cd_transformer_path = os.path.join(
        transformer, "transformer"
    )

    model_cd_transformer_le_path = os.path.join(
        transformer, "transformer_le.joblib"
    )

    cd_transformer_model = Transformer.load(
        model_cd_transformer_path,
        model_cd_transformer_le_path
    )

    logger.info("Model initialized successfully.")


def is_relevant_customer_demand(
    text: str,
    relevant_proba: float,
    cd_logreg_proba: float,
    cd_transformer_proba: float,
):
    """
    Determines if a customer demand prediction should be added to the document.
    cd_logreg_proba is just used as an additional simple out of distribution check,
    we don't care about the label being the same.
    """
    return (
        len(text.split(" ")) > 2
        and cd_logreg_proba > 0.1
        and (
            (relevant_proba > 0.65 and cd_transformer_proba > 0.9)
            or cd_transformer_proba > 0.95
        )
    )


def inference(
    document: json,
    num_cd_predictions: int
):
    # load and process document
    start_time = time.time()
    # document = json.loads(document)[0]
    text = pd.Series(
        content["text"] for content in
        document["contentDomain"]["byId"].values()
    )

    text = pre_processing.clean_text(text)
    latency = time.time() - start_time
    print("Time to load and clean document: {}".format(latency))

    # score relevence model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)[:, 1]
    latency = time.time() - start_time
    print("Time to score relevence model: {}".format(latency))

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = (
        cd_logreg_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd logreg model: {}".format(latency))

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = (
        cd_transformer_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd transformer model: {}".format(latency))

    # format results
    start_time = time.time()
    document_demand_predictions = set()

    for (
        content,
        relevance_prediction,
        cd_logreg_predictions,
        cd_transformer_predictions,
    ) in zip(
        document["contentDomain"]["byId"].values(),
        all_relevance_predictions,
        all_cd_logreg_predictions,
        all_cd_transformer_predictions,
    ):
        if is_relevant_customer_demand(
            content["text"],
            relevance_prediction,
            cd_logreg_predictions[0]["proba"],
            cd_transformer_predictions[0]["proba"],
        ):
            document_demand_predictions.add(
                cd_transformer_predictions[0]["label"]
            )

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    # update document to include document_demand_predictions
    document['documentDemandPredictions'] = list(document_demand_predictions)

    result = document

    latency = time.time() - start_time
    print("Time to format results: {}".format(latency))

    return result


# ===========
# BATCH CHANGE
# ===========
def run(mini_batch):
    """
    Batch Endpoint:
      - mini_batch = list of file paths
      - each file JSON ROOT = document
      - num_preds from env var NUM_PREDS (default 3)

    Output:
      - always returns a valid JSON object with:
          {"predictions": <document>, "metadata": {...}}
      - on error: predictions = input document if parsed, else {}
    """
    results = []

    # NUM_PREDS from env (default 3)
    try:
        num_pred = int(os.getenv("NUM_PREDS", "3"))
    except ValueError:
        num_pred = 3
        logger.warning("NUM_PREDS is not an int. Using %s.", num_pred)

    for item in mini_batch:
        input_document = None

        metadata = {
            "status": "ok",
            "error_type": None,
            "error_message": None,
            "input_file": item,
            "num_preds": num_pred,
        }

        try:
            if not isinstance(item, str) or not os.path.exists(item):
                raise ValueError(f"Input is not a valid file path: {item}")

            with open(item, "r", encoding="utf-8") as f:
                raw_data = f.read()

            logger.info(f"Received request with data from file: {item}")

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Request body cannot be empty!")

            # ROOT=document
            try:
                input_document = json.loads(raw_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format!")

            # minimal sanity check (keeps failures predictable)
            if "contentDomain" not in input_document or "byId" not in input_document.get("contentDomain", {}):
                raise ValueError("Invalid document format (missing contentDomain.byId).")

            response = inference(input_document, num_pred)

            # force JSON-serializable (prevents output with python repr / single quotes)
            response = json.loads(json.dumps(response, default=str))

            results.append({"predictions": response, "metadata": metadata})

        except Exception as e:
            logger.error(f"Error processing file {item}: {str(e)}", exc_info=True)

            metadata["status"] = "error"
            metadata["error_type"] = type(e).__name__
            metadata["error_message"] = str(e)

            # fallback: never lose input if we managed to parse it
            fallback = input_document if isinstance(input_document, dict) else {}
            fallback = json.loads(json.dumps(fallback, default=str))

            results.append({"predictions": fallback, "metadata": metadata})

    return results

