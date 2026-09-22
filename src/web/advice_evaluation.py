"""
Ad-hoc evaluation: does the bot's actual generated advice (the LLM
explanation layer, src/web/llm_explainer.py) hold to the constraints its
own prompt design claims - no invented numbers, no future-percentage
forecasts, no historical fact presented as a live prediction - and is it
plain-English readable? This is the "evaluate the advice the advisor bot
generates" evidence called for by the project brief (exam_resources/
FinalProjectTemplates.pdf, 4.2), separate from and complementary to human
user testing (see docs/user_testing_materials.md), which requires real
participants this script cannot substitute for.

    ./venv/bin/python scripts/evaluate_advice.py

Loads the real persisted production model + feature table (the same ones
the running app uses), samples N_SAMPLES tickers at random (fixed seed
for reproducibility), builds a recommendation for each via
recommendation.build_recommendation, and generates a real explanation via
llm_explainer.generate_explanation - Ollama is called for real (not
mocked), so results reflect actual model output, not the template
fallback. Every explanation is run through all four
src/web/advice_evaluation checks. Prints aggregate pass rates and
readability stats, plus every flagged sentence in full so failures are
inspectable. Saves the per-ticker results to
data/4_experiments/advice_evaluation_results.csv (a dedicated file, not
experiment_log.csv, since this measures explanation-text quality rather
than a model/backtest run and doesn't fit that schema).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random  # noqa: E402
import pandas as pd  # noqa: E402
from src.ai import model_persistence  # noqa: E402
from src.web import recommendation, llm_explainer, advice_evaluation as ev  # noqa: E402

FEATURES_FILE = os.path.join("data", "3_features", "event_driven_features.csv")
N_SAMPLES = 30
SEED = 42
RESULTS_FILE = os.path.join("data", "4_experiments", "advice_evaluation_results.csv")


def main():
    features_df = pd.read_csv(FEATURES_FILE)
    bundle = model_persistence.load_model()
    model = bundle['model']
    feature_cols = bundle['feature_cols']
    spike_threshold = bundle['config'].get('spike_threshold')

    tickers = sorted(features_df['Symbol'].unique())
    random.seed(SEED)
    sample = random.sample(tickers, min(N_SAMPLES, len(tickers)))

    rows = []
    for ticker in sample:
        row = recommendation.find_latest_event_row(features_df, ticker)
        if row is None:
            continue

        rec = recommendation.build_recommendation(
            row, model, feature_cols, spike_threshold=spike_threshold)
        prompt = llm_explainer.format_prompt(rec)
        explanation, source = llm_explainer.generate_explanation(rec)

        unmatched = ev.check_faithfulness(explanation, prompt)
        forecast_leaks = ev.check_forecast_leak(explanation)
        hist_confusions = ev.check_historical_confusion(explanation, rec)
        readability = ev.compute_readability(explanation)

        rows.append({
            'ticker': ticker,
            'as_of_date': rec['as_of_date'],
            'explanation_source': source,
            'explanation': explanation,
            'unmatched_numbers': unmatched,
            'forecast_leaks': forecast_leaks,
            'historical_confusions': hist_confusions,
            'flesch_reading_ease': readability['flesch_reading_ease'],
            'flesch_kincaid_grade': readability['flesch_kincaid_grade'],
        })

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)

    n = len(results_df)
    n_ollama = (results_df['explanation_source'] == 'ollama').sum()
    n_faithful = (results_df['unmatched_numbers'].apply(len) == 0).sum()
    n_no_leak = (results_df['forecast_leaks'].apply(len) == 0).sum()
    n_no_confusion = (results_df['historical_confusions'].apply(len) == 0).sum()
    mean_ease = results_df['flesch_reading_ease'].mean()
    mean_grade = results_df['flesch_kincaid_grade'].mean()

    print(f"Evaluated {n} explanations ({n_ollama} from Ollama, "
          f"{n - n_ollama} from template_fallback)\n")
    print(f"Faithfulness (no invented numbers):     {n_faithful}/{n} "
          f"({n_faithful / n:.0%})")
    print(f"No future-percentage forecast leak:     {n_no_leak}/{n} "
          f"({n_no_leak / n:.0%})")
    print(f"No historical-fact-as-forecast mixup:   {n_no_confusion}/{n} "
          f"({n_no_confusion / n:.0%})")
    print(f"Mean Flesch Reading Ease:               {mean_ease:.1f}")
    print(f"Mean Flesch-Kincaid Grade Level:         {mean_grade:.1f}\n")

    for label, col in (("unmatched numbers", 'unmatched_numbers'),
                        ("forecast leaks", 'forecast_leaks'),
                        ("historical confusions", 'historical_confusions')):
        flagged = results_df[results_df[col].apply(len) > 0]
        if flagged.empty:
            continue
        print(f"--- Examples flagged for {label} ---")
        for _, r in flagged.iterrows():
            print(f"  [{r['ticker']}] {col}={r[col]}")
            print(f"    \"{r['explanation']}\"")
        print()

    print(f"Full results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
