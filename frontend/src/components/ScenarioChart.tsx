import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { Scenario } from "../types";

interface Props {
  scenarios: Scenario[];
}

export default function ScenarioChart({ scenarios }: Props) {
  const data = scenarios.map((s) => ({
    surprise: s.surprise_pct,
    probability: s.spike_probability * 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="surprise"
          stroke="#94a3b8"
          fontSize={11}
          label={{
            value: "Hypothetical Surprise (%)",
            position: "insideBottom",
            offset: -12,
            fill: "#94a3b8",
            fontSize: 11,
          }}
        />
        <YAxis
          domain={[0, 100]}
          stroke="#94a3b8"
          fontSize={11}
          label={{
            value: "Spike probability (%)",
            angle: -90,
            position: "insideLeft",
            fill: "#94a3b8",
            fontSize: 11,
          }}
        />
        <ReferenceLine y={50} stroke="#475569" strokeDasharray="4 4" />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: 8,
          }}
          labelStyle={{ color: "#e2e8f0" }}
          itemStyle={{ color: "#e2e8f0" }}
          formatter={(value) => [
            `${Number(value).toFixed(1)}%`,
            "Spike probability",
          ]}
          labelFormatter={(label) => `Surprise: ${label}%`}
        />
        <Line
          type="monotone"
          dataKey="probability"
          stroke="#60a5fa"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
