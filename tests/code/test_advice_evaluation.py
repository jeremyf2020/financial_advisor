import pytest
from src.web import advice_evaluation as ev


def test_extract_numbers():
    """ Numbers should extract as floats, with any trailing % stripped """
    result = ev.extract_numbers("Confidence: 95%, EPS Estimate: 1.23")

    assert result == [95.0, 1.23]


def test_check_faithfulness_flags_unmatched_number():
    """ A number in the explanation with no match anywhere in the prompt
    (even after rounding) should be flagged as unfaithful """
    prompt = "Model confidence: 95%\nSurprise(%): 5.2"
    explanation = "The model is 70% confident based on the surprise."

    unmatched = ev.check_faithfulness(explanation, prompt)

    assert unmatched == [70.0]


def test_check_faithfulness_allows_rounding():
    """ A number restated at coarser precision than the prompt gave it
    should NOT be flagged - that's faithful rounding, not invention """
    prompt = "Surprise(%): 1.23"
    explanation = "The surprise was about 1.2, roughly 1 in round terms."

    unmatched = ev.check_faithfulness(explanation, prompt)

    assert unmatched == []


def test_check_faithfulness_allows_fraction_to_percent_scale():
    """ A fraction in the prompt (e.g. a 0.07 spike threshold) restated on
    the percentage scale (7%) is a faithful unit conversion, not an
    invented number """
    prompt = "Spike threshold used to define 'Spike': 0.07"
    explanation = "A spike means a move of more than 7%."

    unmatched = ev.check_faithfulness(explanation, prompt)

    assert unmatched == []


def test_check_faithfulness_allows_confidence_complement():
    """ A confidence percentage's complement (100 - x) is directly
    computable from a number the prompt gave, so restating it as "there's
    a 32% chance" when the prompt said "68% confidence" is not invention """
    prompt = "Model confidence: 68%"
    explanation = "There's a 32% chance it could go the other way."

    unmatched = ev.check_faithfulness(explanation, prompt)

    assert unmatched == []


def test_check_faithfulness_ignores_iso_dates():
    """ An ISO date's hyphens must not be misread as negative-number
    signs (e.g. "2026-07-29" as -7, -29), which would make a faithful
    restatement of that same date look like an invented number """
    prompt = "Data as of: 2026-07-29 (most recent earnings event)"
    explanation = "This is based on the earnings event from July 29, 2026."

    unmatched = ev.check_faithfulness(explanation, prompt)

    assert unmatched == []


def test_check_forecast_leak_flags_forward_percentage_without_history():
    """ A forward-looking marker + a specific percentage + no historical
    context in the same sentence should be flagged """
    explanation = "The stock will rise by 25% over the next week."

    flagged = ev.check_forecast_leak(explanation)

    assert len(flagged) == 1
    assert "will rise by 25%" in flagged[0]


def test_check_forecast_leak_allows_historical_percentage():
    """ The same shape of sentence should NOT be flagged once it's
    clearly framed as a historical fact """
    explanation = "Historically, the stock rose by 25% after similar events."

    flagged = ev.check_forecast_leak(explanation)

    assert flagged == []


def test_check_forecast_leak_allows_confidence_percentage():
    """ A percentage describing the model's own confidence, not a
    predicted return magnitude, should not be flagged - "will not
    experience a spike, with 73% confidence" states a class prediction
    and how sure the model is of it, never a stated gain/loss """
    explanation = "The model predicts the stock will not spike, with 73% confidence."

    flagged = ev.check_forecast_leak(explanation)

    assert flagged == []


def test_check_forecast_leak_ignores_unlikely_as_a_forward_marker():
    """ "unlikely" must not match the "likely" marker as a substring -
    that would invert the meaning, treating a hedge that something WON'T
    happen as if it were a forecast that it will """
    explanation = "A large price move of more than 7% is unlikely for this stock."

    flagged = ev.check_forecast_leak(explanation)

    assert flagged == []


def test_check_historical_confusion_flags_forecast_framing_of_past_number():
    """ A historical_outcome number, discussed with forward-looking
    language and no historical marker, should be flagged as presenting a
    known fact as if it were a live forecast """
    rec = {'historical_outcome': {'target_t1_close_ret': 0.03, 'target_t1_high_ret': 0.05}}
    explanation = "The model predicts the stock will rise 3% based on this pattern."

    flagged = ev.check_historical_confusion(explanation, rec)

    assert len(flagged) == 1


def test_check_historical_confusion_allows_historical_framing():
    """ The same historical number, clearly marked as historical/actual,
    should not be flagged even when forward-looking words also appear
    elsewhere in the sentence """
    rec = {'historical_outcome': {'target_t1_close_ret': 0.03, 'target_t1_high_ret': 0.05}}
    explanation = "This is expected based on what actually happened: a 3% return historically."

    flagged = ev.check_historical_confusion(explanation, rec)

    assert flagged == []


def test_count_syllables():
    """ Simple vowel-group counting sanity checks, including the
    trailing-silent-e reduction ("like" -> 'i','e' groups minus the
    silent e = 1) and an all-punctuation token stripping to empty """
    assert ev.count_syllables("cat") == 1
    assert ev.count_syllables("banana") == 3
    assert ev.count_syllables("the") == 1
    assert ev.count_syllables("like") == 1
    assert ev.count_syllables("...") == 0


def test_compute_readability_returns_none_for_empty_text():
    """ Empty/unparseable text should return None scores, not raise """
    result = ev.compute_readability("")

    assert result == {'flesch_reading_ease': None, 'flesch_kincaid_grade': None}


def test_compute_readability_simple_sentence():
    """ A hand-computable case: "The cat sat." - 3 one-syllable words,
    1 sentence, so words_per_sentence=3, syllables_per_word=1 """
    result = ev.compute_readability("The cat sat.")

    assert result['flesch_reading_ease'] == pytest.approx(119.19, abs=0.01)
    assert result['flesch_kincaid_grade'] == pytest.approx(-2.62, abs=0.01)
