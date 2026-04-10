import time

def invoke_with_retry(
    model,
    messages,
    check_fn,
    parse_fn,
    max_retries=3,
    delay=0.5
):
    last_error = None

    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)

            if check_fn(response):
                return parse_fn(response)
            else:
                print(f"[Retry {attempt+1}] Invalid format")

        except Exception as e:
            last_error = e
            print(f"[Retry {attempt+1}] Exception: {e}")

        time.sleep(delay)

    raise ValueError(
        f"Failed after {max_retries} retries. Last error: {last_error}"
    )