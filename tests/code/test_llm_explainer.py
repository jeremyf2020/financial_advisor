import pytest
import requests
from unittest.mock import patch, Mock
from src.web import llm_explainer as llm


def make_rec():
    return {
        'ticker': 'AAPL', 'sector': 'Information Technology', 'as_of_date': '2026-07-30',
        'predicted_label': 'No Spike', 'confidence': 0.947, 'spike_threshold': 0.07,
        'feature_values': {'EPS Estimate': 1.89, 'Reported EPS': 2.02, 'Surprise(%)': 6.74},
        'historical_outcome': {'target_t1_close_ret': -0.0735, 'target_t1_high_ret': -0.0682},
    }


def test_format_prompt_includes_every_rec_value():
    """ Every value in rec should appear in the prompt, so the LLM can't be blamed for missing context """
    rec = make_rec()

    prompt = llm.format_prompt(rec)

    assert rec['ticker'] in prompt
    assert rec['predicted_label'] in prompt
    assert '95' in prompt  # confidence rendered as a percentage (0.947 -> 95%)
    assert '6.74' in prompt
    assert '-0.0735' in prompt


def test_format_prompt_instructs_against_inventing_numbers():
    """ Prompt must explicitly forbid inventing a return percentage the model doesn't predict """
    rec = make_rec()

    prompt = llm.format_prompt(rec)

    assert 'do not' in prompt.lower() or 'not predict' in prompt.lower()


def test_template_fallback_is_deterministic_and_mentions_key_facts():
    rec = make_rec()

    text = llm.template_fallback(rec)

    assert 'AAPL' in text
    assert 'No Spike' in text
    assert '95%' in text  # 0.947 rounds to 95%


def test_call_ollama_posts_expected_request_shape():
    """ Should POST to /api/generate with stream=False and return the parsed response text """
    mock_response = Mock()
    mock_response.json.return_value = {"response": "  Here is your advice.  "}
    mock_response.raise_for_status.return_value = None

    with patch('src.web.llm_explainer.requests.post', return_value=mock_response) as mock_post:
        text = llm.call_ollama("some prompt", model_name="llama3.2", host="http://localhost:11434")

    assert text == "Here is your advice."
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": "some prompt", "stream": False},
        timeout=20)


def test_generate_explanation_uses_ollama_when_available():
    """ Should return the Ollama response and mark the source as 'ollama' """
    rec = make_rec()

    with patch('src.web.llm_explainer.call_ollama', return_value="LLM-generated prose") as mock_call:
        text, source = llm.generate_explanation(rec)

    assert text == "LLM-generated prose"
    assert source == "ollama"
    mock_call.assert_called_once()


def test_generate_explanation_falls_back_when_ollama_unreachable():
    """ A connection error should silently fall back to template text, never raise """
    rec = make_rec()

    with patch('src.web.llm_explainer.call_ollama',
               side_effect=requests.exceptions.ConnectionError("refused")):
        text, source = llm.generate_explanation(rec)

    assert source == "template_fallback"
    assert 'AAPL' in text
