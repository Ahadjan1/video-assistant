import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL")  # optional

# 1 request har 5 second, free tier ki limit se bachne ke liye
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def _make(model: str, temperature: float):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        rate_limiter=_rate_limiter,
    )


def get_llm(temperature: float = 0.3):
    # 503 / server errors par 5 baar dobara koshish, har baar thoda zyada intezaar
    llm = _make(GEMINI_MODEL, temperature).with_retry(
        retry_if_exception_type=(GoogleAPIError,),
        stop_after_attempt=5,
        wait_exponential_jitter=True,
    )
    if GEMINI_FALLBACK_MODEL:
        backup = _make(GEMINI_FALLBACK_MODEL, temperature)
        return llm.with_fallbacks([backup])
    return llm