def run(mini_batch):
    """
    Batch Endpoint
    - ALWAYS returns JSON (string, double quotes)
    - JSON Lines (1 JSON per input file)
    """

    results = []

    # NUM_PREDS from env (default 3)
    try:
        num_pred = int(os.getenv("NUM_PREDS", "3"))
    except ValueError:
        num_pred = 3

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

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Request body cannot be empty!")

            input_document = json.loads(raw_data)

            if "contentDomain" not in input_document or "byId" not in input_document.get("contentDomain", {}):
                raise ValueError("Invalid document format (missing contentDomain.byId).")

            response = inference(input_document, num_pred)

            # HARD JSON GUARANTEE
            result_obj = {
                "predictions": response,
                "metadata": metadata,
            }

        except Exception as e:
            metadata["status"] = "error"
            metadata["error_type"] = type(e).__name__
            metadata["error_message"] = str(e)

            fallback = input_document if isinstance(input_document, dict) else {}

            result_obj = {
                "predictions": fallback,
                "metadata": metadata,
            }

        # 🔒 JEDYNE MIEJSCE SERIALIZACJI
        results.append(
            json.dumps(
                result_obj,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    # JSONL: List[str]
    return results
