'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface Segment {
  segment_name: string
  count: number
  avg_clv: number
  avg_churn_prob: number
}

interface SegmentsChartProps {
  segments: Segment[]
}

export default function SegmentsChart({ segments }: SegmentsChartProps) {
  if (!segments || segments.length === 0) {
    return <p className="text-gray-400 text-center py-8">No segment data available</p>
  }

  const data = segments.map(seg => ({
    name: seg.segment_name.split(' ').slice(0, 2).join(' '),
    customers: seg.count,
    avgCLV: Math.round(seg.avg_clv),
    churnRisk: Math.round(seg.avg_churn_prob * 100)
  }))

  return (
    <div className="space-y-6">
      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" fontSize={12} stroke="#9CA3AF" />
          <YAxis stroke="#9CA3AF" />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#F3F4F6' }}
            itemStyle={{ color: '#60A5FA' }}
          />
          <Legend wrapperStyle={{ color: '#9CA3AF' }} />
          <Bar dataKey="customers" fill="#3b82f6" name="Customers" />
        </BarChart>
      </ResponsiveContainer>

      {/* Segment Cards */}
      <div className="grid grid-cols-1 gap-3">
        {segments.map((seg, index) => (
          <div
            key={index}
            className="border border-gray-700 bg-gray-750 rounded-lg p-4 hover:bg-gray-700 transition-colors"
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-white">{seg.segment_name}</h3>
                <p className="text-sm text-gray-400 mt-1">
                  {seg.count.toLocaleString()} customers
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-400">Avg CLV</p>
                <p className="text-lg font-bold text-green-400">
                  ${seg.avg_clv.toFixed(0)}
                </p>
              </div>
            </div>
            <div className="mt-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Churn Risk</span>
                <span className={`font-semibold ${
                  seg.avg_churn_prob > 0.6 ? 'text-red-400' :
                  seg.avg_churn_prob > 0.3 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {(seg.avg_churn_prob * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 w-full bg-gray-600 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    seg.avg_churn_prob > 0.6 ? 'bg-red-500' :
                    seg.avg_churn_prob > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${seg.avg_churn_prob * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
