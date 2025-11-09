import os
import time
from typing import Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Optional import (only if Streamlit exists)
try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()


def _get_secret(name: str) -> str | None:
    """Get environment variable from Streamlit secrets or OS env."""
    val = None
    if st and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)
        except Exception:
            pass
    return val or os.getenv(name)


class Chain:
    def __init__(self):
        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not set in Streamlit Secrets or .env")

        # ✅ List both models here
        self.models = [
            "llama-3.1-8b-instant",       # ✅ Fast, smaller model
            "llama-3.3-70b-versatile"     # ✅ Backup: larger, better model
        ]
        self.api_key = api_key
        self._make_llm(self.models[0])

    def _make_llm(self, model_name: str):
        """Initialize the model (handle both param names)."""
        try:
            self.llm = ChatGroq(api_key=self.api_key, model_name=model_name, temperature=0)
        except TypeError:
            self.llm = ChatGroq(groq_api_key=self.api_key, model_name=model_name, temperature=0)

    def _invoke_with_retry(self, chain, payload):
        """Retry once if rate limit occurs."""
        for i in range(2):
            try:
                return chain.invoke(payload)
            except Exception as e:
                msg = str(e).lower()
                if "rate limit" in msg or "429" in msg:
                    time.sleep(3)
                else:
                    raise
        raise

    def extract_jobs(self, cleaned_text: str) -> List[dict[str, Any]]:
        """Extract job details from scraped text."""
        prompt_extract = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
The scraped text is from a careers page.
Extract job postings and return valid JSON with keys: role, experience, skills, description.
Return ONLY JSON (no extra text)."""
        )

        for model in self.models:
            self._make_llm(model)
            chain_extract = prompt_extract | self.llm
            try:
                res = self._invoke_with_retry(chain_extract, {"page_data": cleaned_text})
                parsed = JsonOutputParser().parse(getattr(res, "content", res))
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception as e:
                last_error = e
                continue
        raise last_error

    def write_mail(self, job: dict, links: list[str]) -> str:
        """Generate personalized cold email."""
        prompt_email = PromptTemplate.from_template(
            """### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, BDE at AtliQ (AI & Software Consulting).
Write a short, tailored cold email to the company.
Mention relevant portfolio links: {link_list}.
Do NOT include any preamble — only return the email in Markdown."""
        )

        for model in self.models:
            self._make_llm(model)
            chain_email = prompt_email | self.llm
            try:
                res = self._invoke_with_retry(chain_email, {"job_description": str(job), "link_list": links})
                return getattr(res, "content", str(res))
            except Exception:
                continue

        raise RuntimeError("⚠️ All Groq models failed due to rate limits or network errors.")
