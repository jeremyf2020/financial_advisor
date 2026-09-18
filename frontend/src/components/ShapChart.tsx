import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";

interface Props {
  shapValues: Record<string, number>;
}

export default function ShapChart({ shapValues }: Props) {
  const data = Object.entries(shapValues)
    .map(([feature, value]) => ({ feature, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            horizontal={false}
          />
          <XAxis type="number" stroke="#94a3b8" fontSize={11} />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#94a3b8"
            fontSize={11}
            width={110}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 8,
            }}
            labelStyle={{ color: "#e2e8f0" }}
            itemStyle={{ color: "#e2e8f0" }}
            formatter={(value) => Number(value).toFixed(4)}
          />
          <Bar dataKey="value">
            {data.map((entry) => (
              <Cell
                key={entry.feature}
                fill={entry.value >= 0 ? "#34d399" : "#f87171"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-500 mt-1">
        <span className="text-emerald-400">■</span> pushed toward Spike ·{" "}
        <span className="text-red-400">■</span> pushed toward No Spike
      </p>
    </div>
  );
}
