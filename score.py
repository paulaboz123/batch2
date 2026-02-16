# score.py (Azure ML Batch Endpoint) — end-to-end
# Założenia zgodne z Twoimi ustaleniami:
# 1) INPUT: każdy plik wejściowy .json ma ROOT = dokument (bez wrappera {"document":..., "num_preds":...})
# 2) num_preds pochodzi z env var: NUM_PREDS (default: 3)
# 3) OUTPUT: zawsze poprawny JSON; zawsze zwracamy dokument (wynik inference lub wejściowy fallback)
#    + metadata (status/error) — nigdy nie zwracamy "gołego" {"error":...} zamiast dokumentu
# 4) Nie zmieniamy logiki init()/inference()/is_relevant_customer_demand() poza niezbędnymi importami;
#    jedyne zmiany logiczne są w run() (batch contract + error handling + stały output)

import os
import time
import json
import logging
import pandas as pd

from models.logistic_regression import LogisticRegression  # noqa
from models.transformer import Transformer  # noqa
import pre_processing  # noqa

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
    if not model:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    # debug jak w Twoim oryginale (zostawiam)
    print(f"line 23: {os.listdir(model)}")
    print(f"line 24: {model}")

    # struktura modelu jak w Twoim kodzie
    model = os.path.join(model, "albot")
    print(f"line 27: {os.listdir(model)}")

    relevance_model = LogisticRegression.load(
        os.path.join(model, "logreg_relevance.joblib")
    )

    cd_logreg_model = LogisticRegression.load(
        os.path.join(model, "logreg_cd.joblib")
    )

    transformer = os.path.join(model, "transformer_model")

    model_cd_transformer_path = os.path.join(transformer, "transformer")
    model_cd_transformer_le_path = os.path.join(transformer, "transformer_le.joblib")

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


def inference(document: json, num_cd_predictions: int):
    # load and process document
    start_time = time.time()

    text = pd.Series(
        content["text"]
        for content in document["contentDomain"]["byId"].values()
    )

    text = pre_processing.clean_text(text)
    latency = time.time() - start_time
    print("Time to load and clean document: {}".format(latency))

    # score relevance model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)[:, 1]
    latency = time.time() - start_time
    print("Time to score relevence model: {}".format(latency))

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = cd_logreg_model.predict_top_n_labels_with_proba(
        text, num_cd_predictions
    )
    latency = time.time() - start_time
    print("Time to score cd logreg model: {}".format(latency))

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = cd_transformer_model.predict_top_n_labels_with_proba(
        text, num_cd_predictions
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
            document_demand_predictions.add(cd_transformer_predictions[0]["label"])

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    document["documentDemandPredictions"] = list(document_demand_predictions)

    latency = time.time() - start_time
    print("Time to format results: {}".format(latency))

    return document


def run(mini_batch):
    """
    Azure ML Batch Endpoint contract:
      - mini_batch: list of file paths
      - each file content is JSON where ROOT is the document

    num_preds:
      - taken from env var NUM_PREDS (default 3)

    Output:
      - always returns a JSON-serializable object
      - always includes the document (in predictions)
      - always includes metadata with status and error info
    """
    results = []

    # num_preds from env
    num_preds_env = os.getenv("NUM_PREDS", "3")
    try:
        num_pred = int(num_preds_env)
    except ValueError:
        num_pred = 3
        logger.warning(f"NUM_PREDS is not an int: {num_preds_env}. Using {num_pred}.")

    for item in mini_batch:
        start = time.time()
        input_document = None

        metadata = {
            "status": "ok",          # ok | error
            "error_type": None,
            "error_message": None,
            "input_file": item,
            "num_preds": num_pred,
            "processing_ms": None,
        }

        try:
            if not isinstance(item, str) or not os.path.exists(item):
                raise ValueError(f"Input is not a valid file path: {item}")

            with open(item, "r", encoding="utf-8") as f:
                raw_data = f.read()

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Request body cannot be empty!")

            # Parse JSON document (root)
            try:
                input_document = json.loads(raw_data)
            except json.JSONDecodeError as je:
                raise ValueError(f"Invalid JSON format: {str(je)}")

            # Minimal structure check (optional but helps catch bad inputs early)
            if (
                "contentDomain" not in input_document
                or "byId" not in input_document.get("contentDomain", {})
            ):
                raise ValueError("Invalid document format (missing contentDomain.byId).")

            # Inference
            output_document = inference(input_document, num_pred)

            # Ensure JSON output (prevents accidental Python repr with single quotes)
            output_document = json.loads(json.dumps(output_document, default=str))

            metadata["processing_ms"] = int((time.time() - start) * 1000)

            results.append(
                {
                    "predictions": output_document,
                    "metadata": metadata,
                }
            )

        except Exception as e:
            logger.error(f"Error processing item {item}: {str(e)}", exc_info=True)

            metadata["status"] = "error"
            metadata["error_type"] = type(e).__name__
            metadata["error_message"] = str(e)
            metadata["processing_ms"] = int((time.time() - start) * 1000)

            # Fallback: always return a document-like JSON.
            # If parsing succeeded, return the original input doc (no data loss).
            # If parsing failed, return {}.
            fallback_document = input_document if isinstance(input_document, dict) else {}
            fallback_document = json.loads(json.dumps(fallback_document, default=str))

            results.append(
                {
                    "predictions": fallback_document,
                    "metadata": metadata,
                }
            )

    return results
