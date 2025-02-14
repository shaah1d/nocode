import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

const DataVisualization: React.FC = () => {
  // Define chart data
  const chartData = [
    { name: "ROC-AUC", value: 0.86 },
    { name: "Precision", value: 0.90 },
    { name: "Recall", value: 0.90 },
    { name: "F1 Score", value: 0.86 },
  ];

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4 text-center">ML Model Performance Metrics</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <XAxis dataKey="name" className="text-sm" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#3498db" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DataVisualization;
