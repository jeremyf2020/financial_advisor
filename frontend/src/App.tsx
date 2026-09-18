import { useState, useEffect } from "react";
import TickerSearch from "./components/TickerSearch";
import RecommendationCard from "./components/RecommendationCard";
import { getTickers, getRecommendation } from "./api";
import type { Ticker, Recommendation } from "./types";

export default function App() {
  const [tickers, setTickers] = useState<Ticker[]>([]);
  const [result, setResult] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTickers()
      .then(setTickers)
      .catch(() => setTickers([]));
  }, []);

  async function handleSubmit(ticker: string) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await getRecommendation(ticker);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="border-b border-slate-700 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center gap-3">
          <span className="text-2xl">🤖</span>
          <div>
            <h1 className="text-lg font-bold text-slate-100 leading-tight m-0">
              Financial Advisor Bot
            </h1>
            <p className="text-xs text-slate-400 m-0">
              Earnings-event-driven recommendations · XGBoost
            </p>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-8 space-y-6">
        <TickerSearch
          tickers={tickers}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {loading && (
          <div className="flex items-center justify-center h-32 text-slate-400">
            <div className="text-center">
              <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p>Fetching recommendation...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        {result && <RecommendationCard result={result} />}

        <footer className="text-xs text-slate-500 pt-4">
          Prototype — predictions are based on a model still under active tuning
          and should not be used for real investment decisions.
        </footer>
      </main>
    </div>
  );
}
