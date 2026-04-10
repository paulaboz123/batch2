def run(mini_batch):
    """
    Batch Endpoint
    - ALWAYS returns JSON (string, double quotes)
    - JSON Lines (1 JSON per input file)
    """

    batch_start = time.time()
    results = []

    # NUM_PREDS from env (default 3)
    try:
        num_pred = int(os.getenv("NUM_PREDS", "3"))
    except ValueError:
        num_pred = 3

    for item in mini_batch:
        file_start = time.time()
        input_document = None

        metadata = {
            "status": "ok",
            "error_type": None,
            "error_message": None,
            "input_file": item,
            "num_preds": num_pred,
        }

        try:
            # Validate input path
            if not isinstance(item, str) or not os.path.exists(item):
                raise ValueError(f"Input is not a valid file path: {item}")

            # Read file
            with open(item, "r", encoding="utf-8") as f:
                raw_data = f.read()

            if not raw_data or raw_data.strip() == "":
                raise ValueError("Request body cannot be empty!")

            # Parse JSON
            input_document = json.loads(raw_data)

            # Validate structure
            if "contentDomain" not in input_document or "byId" not in input_document.get("contentDomain", {}):
                raise ValueError("Invalid document format (missing contentDomain.byId).")

            # Run inference
            response, inference_latency = inference(input_document, num_pred)

            file_latency = time.time() - file_start

            # Log success
            logger.info(
                "Inference success",
                extra={
                    "custom_dimensions": {
                        "success": True,
                        "file_name": os.path.basename(item),
                        "latency_sec": file_latency,
                        "inference_latency_sec": inference_latency
                    }
                }
            )

            result_obj = {
                "predictions": response,
                "metadata": metadata,
            }

        except Exception as e:
            file_latency = time.time() - file_start

            metadata["status"] = "error"
            metadata["error_type"] = type(e).__name__
            metadata["error_message"] = str(e)

            fallback = input_document if isinstance(input_document, dict) else {}

            # Log error
            logger.error(
                "Inference failed",
                exc_info=True,
                extra={
                    "custom_dimensions": {
                        "success": False,
                        "file_name": os.path.basename(item),
                        "latency_sec": file_latency,
                        "error_type": type(e).__name__
                    }
                }
            )

            result_obj = {
                "predictions": fallback,
                "metadata": metadata,
            }

        # Serialize result (JSONL)
        results.append(
            json.dumps(
                result_obj,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    batch_latency = time.time() - batch_start

    # Log batch summary
    logger.info(
        "Batch completed",
        extra={
            "custom_dimensions": {
                "batch_size": len(mini_batch),
                "batch_latency_sec": batch_latency
            }
        }
    )

    return results
