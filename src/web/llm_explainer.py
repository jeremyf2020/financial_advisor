import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def format_prompt(rec):
    """
    Pure: builds a prompt that only references values already present in
    `rec`, with explicit instructions not to invent numbers - so a
    hallucinated number can be traced straight back to a prompt-adherence
    failure, not "the model made up new data".
    """
    features = "\n".join(f"  - {k}: {v}" for k, v in rec['feature_values'].items())
    historical = rec['historical_outcome']

    return (
        "You are a financial advisor bot explaining a stock prediction to a "
        "non-technical user in 2-3 plain-English sentences.\n\n"
        "Use ONLY the numbers given below - do not invent, estimate, or round "
        "any number that isn't explicitly provided.\n\n"
        f"Ticker: {rec['ticker']}\n"
        f"Sector: {rec['sector']}\n"
        f"Data as of: {rec['as_of_date']} (this is the most recent earnings "
        "event on file, not necessarily today)\n"
        f"Model prediction: {rec['predicted_label']}\n"
        f"Model confidence: {rec['confidence']:.0%}\n"
        f"Spike threshold used to define 'Spike': {rec['spike_threshold']}\n"
        f"Feature values used by the model:\n{features}\n\n"
        "IMPORTANT: the model only predicts whether a spike happens or not - "
        "it does NOT predict a percentage return. Do not state or imply a "
        "specific future percentage gain or loss.\n\n"
        "For reference only, here is what ACTUALLY happened after this "
        "specific past earnings event (a historical fact, not a forecast) - "
        f"you may mention it as historical context but must not present it as "
        f"a prediction: close price return was "
        f"{historical['target_t1_close_ret']}, high price return was "
        f"{historical['target_t1_high_ret']}."
    )


def template_fallback(rec):
    """
    Pure: deterministic plain-English text built with an f-string, no LLM
    involved. Used when Ollama is unreachable so the app still works
    end-to-end on a machine without it installed/running.
    """
    return (
        f"Based on the earnings event for {rec['ticker']} on {rec['as_of_date']}, "
        f"the model predicts '{rec['predicted_label']}' with {rec['confidence']:.0%} "
        "confidence. This reflects the most recent earnings data on file for "
        f"{rec['ticker']}, not necessarily today's market conditions."
    )


def call_ollama(prompt, model_name=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=20):
    """
    I/O: POST to Ollama's local REST API (/api/generate, stream=False).
    Raises requests.exceptions.RequestException on any failure - the
    caller (generate_explanation) decides the fallback.
    """
    response = requests.post(
        f"{host}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=timeout)
    response.raise_for_status()

    return response.json()["response"].strip()


def generate_explanation(rec, model_name=OLLAMA_MODEL, host=OLLAMA_HOST):
    """
    Orchestrator: format_prompt -> call_ollama, falling back to
    template_fallback on any network/timeout/HTTP error. Returns
    (text, source) where source is "ollama" or "template_fallback", so
    callers can show which path actually produced the text.
    """
    prompt = format_prompt(rec)

    try:
        return call_ollama(prompt, model_name, host), "ollama"
    except requests.exceptions.RequestException:
        return template_fallback(rec), "template_fallback"
