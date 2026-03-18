import logging
import functools
import time

perf_logger = logging.getLogger("treerage.perf")
def track_time(name=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            
            try:
                return fn(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                perf_logger.info({
                    "type": "function",
                    "name": name or fn.__name__,
                    "duration": round(duration,2)
                })
            
        return wrapper
    return decorator    

def track_llm_call(fn):
    @functools.wraps(fn)
    def wrapper(*args, llm_label=None, **kwargs):
        start = time.perf_counter()
        label = llm_label or fn.__name__
        try:
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start
            usage = result["usage"]
        
            perf_logger.info({
                "type": "llm",
                "name": label,
                "duration": round(duration,2),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cached_tokens": getattr(getattr(usage, "prompt_tokens_details"), "cached_tokens", None)
            })
        
            return result

        except Exception as e:
            duration = time.perf_counter() - start
            perf_logger.info({
                "type": "llm_error",
                "name": llm_label,
                "duration": round(duration,2),
                "error": str(e)
            })
            raise
    return wrapper

def patch_llm_call(client):
    original = client.chat_json
    wrapped = track_llm_call(original)
    client.chat_json = wrapped