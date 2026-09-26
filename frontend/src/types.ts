export interface Ticker {
  symbol: string;
  sector: string | null;
}

export interface HistoricalOutcome {
  target_t1_close_ret: number | null;
  target_t1_high_ret: number | null;
}

export interface Recommendation {
  ticker: string;
  as_of_date: string;
  sector: string | null;
  predicted_class: number;
  predicted_label: string;
  confidence: number;
  spike_threshold: number | null;
  feature_values: Record<string, number | null>;
  historical_outcome: HistoricalOutcome;
  explanation: string;
  explanation_source: string;
  shap_values: Record<string, number>;
}

export interface UpcomingEarning {
  symbol: string;
  date: string;
  sector: string | null;
  eps_estimate: number | null;
}

export interface Scenario {
  surprise_pct: number;
  reported_eps: number;
  predicted_label: string;
  spike_probability: number;
}

export interface ScenarioAnalysis {
  ticker: string;
  upcoming_earnings_date: string;
  as_of_date: string;
  sector: string | null;
  eps_estimate: number;
  spike_threshold: number | null;
  context_features: Record<string, number | null>;
  scenarios: Scenario[];
}
