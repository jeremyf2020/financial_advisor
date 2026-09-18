import type { Recommendation } from "../types";
import ShapChart from "./ShapChart";

function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return "n/a";
  return `${(value * 100).toFixed(2)}%`;
}

interface Props {
  result: Recommendation;
}

export default function RecommendationCard({ result }: Props) {
  const isSpike = result.predicted_label === "Spike";

  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold text-slate-100 m-0">
          {result.ticker}
        </h2>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide ${
            isSpike
              ? "bg-emerald-500/15 text-emerald-400"
              : "bg-slate-700 text-slate-300"
          }`}
        >
          {result.predicted_label}
        </span>
      </div>

      <p className="text-slate-300">
        Confidence:{" "}
        <span className="font-semibold text-blue-400">
          {(result.confidence * 100).toFixed(0)}%
        </span>
        <span className="text-slate-500 text-sm">
          {" "}
          (spike threshold: {result.spike_threshold})
        </span>
      </p>

      <div className="bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm rounded-lg px-3 py-2">
        📅 Data as of <strong>{result.as_of_date}</strong> — the most recent
        earnings event on file for {result.ticker}, not necessarily today.
      </div>

      <p className="text-slate-300 leading-relaxed">
        {result.explanation}{" "}
        <span className="text-slate-500 text-xs">
          ({result.explanation_source})
        </span>
      </p>

      <div>
        <p className="text-slate-300 font-medium mb-1">
          Why did the model decide this?
        </p>
        <ShapChart shapValues={result.shap_values} />
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Feature values used by the model
        </summary>
        <table className="w-full mt-2 border-collapse">
          <tbody>
            {Object.entries(result.feature_values).map(([key, value]) => (
              <tr key={key} className="border-b border-slate-700/60">
                <td className="py-1.5 text-slate-400">{key}</td>
                <td className="py-1.5 text-slate-200 text-right">
                  {value === null ? "n/a" : value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <details className="text-sm">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Historical outcome (already-realized fact, not a forecast)
        </summary>
        <p className="mt-2 text-slate-400">
          What <strong className="text-slate-200">actually happened</strong>{" "}
          after this specific past earnings event — the model does not predict
          this number.
        </p>
        <table className="w-full mt-2 border-collapse">
          <tbody>
            <tr className="border-b border-slate-700/60">
              <td className="py-1.5 text-slate-400">T+1 Close return</td>
              <td className="py-1.5 text-slate-200 text-right">
                {formatPercent(result.historical_outcome.target_t1_close_ret)}
              </td>
            </tr>
            <tr>
              <td className="py-1.5 text-slate-400">T+1 High return</td>
              <td className="py-1.5 text-slate-200 text-right">
                {formatPercent(result.historical_outcome.target_t1_high_ret)}
              </td>
            </tr>
          </tbody>
        </table>
      </details>
    </div>
  );
}
