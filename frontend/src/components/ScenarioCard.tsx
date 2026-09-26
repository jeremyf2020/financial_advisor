import type { ScenarioAnalysis } from "../types";
import ScenarioChart from "./ScenarioChart";

interface Props {
  result: ScenarioAnalysis;
}

export default function ScenarioCard({ result }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold text-slate-100 m-0">
          {result.ticker}
        </h2>
        <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide bg-blue-500/15 text-blue-400">
          Upcoming Earnings Scenario
        </span>
      </div>

      <div className="bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm rounded-lg px-3 py-2">
        📅 Earnings scheduled <strong>{result.upcoming_earnings_date}</strong>,
        not yet reported. This is <strong>not a forecast</strong> of what will
        happen — the actual surprise is unknown until then. The chart below
        shows, for each <em>hypothetical</em> surprise outcome, what the model
        would say, using {result.ticker}'s real momentum/sector-rank data as of{" "}
        <strong>{result.as_of_date}</strong> and an analyst EPS estimate of{" "}
        {result.eps_estimate.toFixed(2)}.
      </div>

      <ScenarioChart scenarios={result.scenarios} />

      <details className="text-sm">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Fixed context used for every scenario
        </summary>
        <table className="w-full mt-2 border-collapse">
          <tbody>
            {Object.entries(result.context_features).map(([key, value]) => (
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
    </div>
  );
}
