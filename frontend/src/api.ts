import type { Ticker, Recommendation } from "./types";

export async function getTickers(): Promise<Ticker[]> {
  const response = await fetch("/api/tickers");
  if (!response.ok) {
    throw new Error(`Failed to load tickers (${response.status})`);
  }
  return response.json();
}

export async function getRecommendation(
  ticker: string,
): Promise<Recommendation> {
  const response = await fetch(`/api/recommend/${encodeURIComponent(ticker)}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}
