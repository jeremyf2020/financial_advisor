import type { UpcomingEarning } from "../types";

interface Props {
  earnings: UpcomingEarning[];
  onSelect: (ticker: string) => void;
  loading: boolean;
}

export default function UpcomingEarnings({
  earnings,
  onSelect,
  loading,
}: Props) {
  if (earnings.length === 0) return null;

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700">
        <h2 className="text-sm font-semibold text-slate-200 m-0">
          Upcoming Earnings
        </h2>
        <p className="text-xs text-slate-500 mt-1 m-0">
          Scheduled, not yet reported — click a ticker to see how the model's
          call would shift across a range of possible surprise outcomes.
        </p>
      </div>
      <ul className="divide-y divide-slate-700/60 max-h-64 overflow-y-auto">
        {earnings.map((e) => (
          <li key={e.symbol}>
            <button
              onClick={() => onSelect(e.symbol)}
              disabled={loading}
              className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-slate-700/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <span className="text-slate-100 font-medium w-16">
                {e.symbol}
              </span>
              <span className="text-slate-500 text-xs flex-1 truncate">
                {e.sector}
              </span>
              <span className="text-slate-400 text-sm">{e.date}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
