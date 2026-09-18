import { useState, type FormEvent } from "react";
import type { Ticker } from "../types";

interface Props {
  tickers: Ticker[];
  onSubmit: (ticker: string) => void;
  loading: boolean;
}

export default function TickerSearch({ tickers, onSubmit, loading }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (value.trim()) {
      onSubmit(value.trim());
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        list="ticker-options"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="Enter a ticker, e.g. AAPL"
        autoComplete="off"
        className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <datalist id="ticker-options">
        {tickers.map((t) => (
          <option key={t.symbol} value={t.symbol}>
            {t.sector}
          </option>
        ))}
      </datalist>
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="px-5 py-2.5 rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Loading…" : "Get Advice"}
      </button>
    </form>
  );
}
