# chains.py
import os
import time
from typing import Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Optional import (only if running under Streamlit)
try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()


def _get_secret(name: str) -> str | None:
    """Read from Streamlit secrets (if present) or OS env."""
    val = None
    if st and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)
        except Exception:
            pass
    return val or os.getenv(name)


def _make_llm(api_key: str, model_name: str, temperature: float = 0.0):
    """
    Build a ChatGroq LLM. Handles both param names across versions.
    Add max_tokens if your installed version supports it; if not, it's ignored.
    """
    kwargs = dict(model_name=model_name, temperature=temperature)
    # Uncomment the next line if your version supports max_tokens (many do):
    # kwargs["max_tokens"] = 600
    try:
        return ChatGroq(api_key=api_key, **kwargs)
    except TypeError:
        return ChatGroq(groq_api_key=api_key, **kwargs)


def _invoke_with_retry(chain, payload, retries: int = 1, backoff_sec: float = 3.0):
    """
    Retry for transient rate-limit (429) or throttling messages.
    """
    last = None
    for _ in range(retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e).lower()
            last = e
            if ("rate limit" in msg) or ("429" in msg) or ("rate_limit_exceeded" in msg):
                time.sleep(backoff_sec)
                continue
            raise
    raise last


class Chain:
    """
    - extract_jobs(...) → prefers 8B, falls back to 70B
    - write_mail(...)  → prefers 70B, falls back to 8B
    You can also set separate keys to isolate quotas:
        GROQ_API_KEY_FAST   -> used for 8B
        GROQ_API_KEY_HEAVY  -> used for 70B
    If these aren't set, we use GROQ_API_KEY for both.
    """

    def __init__(self):
        # Base / fallback key
        base_key = _get_secret("GROQ_API_KEY")
        if not base_key:
            raise ValueError("❌ GROQ_API_KEY not set in Streamlit Secrets or environment.")

        # Optional per-model keys (lets you separate quotas)
        self.fast_key = _get_secret("GROQ_API_KEY_FAST") or base_key
        self.heavy_key = _get_secret("GROQ_API_KEY_HEAVY") or base_key

        # Model choices
        self.fast_model = "llama-3.1-8b-instant"
        self.heavy_model = "llama-3.3-70b-versatile"

        # Parsers
        self._json_parser = JsonOutputParser()

    # -------------------- EXTRACT --------------------
    def extract_jobs(self, cleaned_text: str) -> List[dict[str, Any]]:
        """
        Prefer small/fast model (8B) for extraction; fallback to 70B if needed.
        """
        prompt = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
The scraped text is from a careers page.
Extract job postings and return valid JSON with keys: role, experience, skills, description.
Return ONLY JSON (no extra text)."""
        )

        # Try fast first, then heavy
        attempts = [
            (self.fast_key, self.fast_model),
            (self.heavy_key, self.heavy_model),
        ]

        last_error = None
        for api_key, model in attempts:
            llm = _make_llm(api_key=api_key, model_name=model, temperature=0)
            chain = prompt | llm
            try:
                res = _invoke_with_retry(chain, {"page_data": cleaned_text}, retries=1, backoff_sec=3.0)
                content = getattr(res, "content", res)
                parsed = self._json_parser.parse(content)
                return parsed if isinstance(parsed, list) else [parsed]
            except (OutputParserException, Exception) as e:
                last_error = e
                continue

        # If both fail:
        raise last_error or RuntimeError("Extraction failed.")

    # -------------------- WRITE EMAIL --------------------
    def write_mail(self, job: dict, links: List[str]) -> str:
        """
        Prefer large/quality model (70B) for writing; fallback to 8B if needed.
        """
        prompt = PromptTemplate.from_template(
            """### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, BDE at AtliQ (AI & Software Consulting).
Write a short, tailored cold email to the company.
Mention relevant portfolio links: {link_list}.
Do NOT include any preamble — only return the email in Markdown."""
        )

        # Try heavy first, then fast
        attempts = [
            (self.heavy_key, self.heavy_model),
            (self.fast_key, self.fast_model),
        ]

        last_error = None
        for api_key, model in attempts:
            llm = _make_llm(api_key=api_key, model_name=model, temperature=0)
            chain = prompt | llm
            try:
                res = _invoke_with_retry(
                    chain,
                    {"job_description": str(job), "link_list": links},
                    retries=1,
                    backoff_sec=3.0,
                )
                return getattr(res, "content", str(res))
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("Email writing failed.")
