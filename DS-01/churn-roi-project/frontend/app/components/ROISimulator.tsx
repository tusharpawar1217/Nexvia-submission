'use client'

import { useState } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Segment {
  segment_name: string
}

interface ROISimulatorProps {
  segments: Segment[]
}

export default function ROISimulator({ segments }: ROISimulatorProps) {
  const [campaignCost, setCampaignCost] = useState('50')
  const [retentionRate, setRetentionRate] = useState('30')
  const [targetSegment, setTargetSegment] = useState('all')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const calculateROI = async () => {
    // Validate inputs
    const cost = parseFloat(campaignCost)
    const rate = parseFloat(retentionRate)
    
    if (isNaN(cost) || cost <= 0) {
      alert('Please enter a valid campaign cost greater than 0')
      return
    }
    
    if (isNaN(rate) || rate < 0 || rate > 100) {
      alert('Please enter a valid retention rate between 0 and 100')
      return
    }
    
    setLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/roi`, {
        campaign_cost: cost,
        retention_rate: rate / 100,
        target_segment: targetSegment === 'all' ? null : targetSegment
      })
      setResult(response.data)
    } catch (error) {
      console.error('Error calculating ROI:', error)
      alert('Failed to calculate ROI. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Campaign Cost per Customer ($)
          </label>
          <input
            type="number"
            value={campaignCost}
            onChange={(e) => setCampaignCost(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="50"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Expected Retention Rate (%)
          </label>
          <input
            type="number"
            value={retentionRate}
            onChange={(e) => setRetentionRate(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="30"
            min="0"
            max="100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Target Segment
          </label>
          <select
            value={targetSegment}
            onChange={(e) => setTargetSegment(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All High-Risk Customers</option>
            {segments.map((seg, index) => (
              <option key={index} value={seg.segment_name}>
                {seg.segment_name}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={calculateROI}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
        >
          {loading ? 'Calculating...' : 'Calculate ROI'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="border-t border-gray-700 pt-6 space-y-4">
          <h3 className="font-semibold text-lg text-white">Results</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
              <p className="text-sm text-gray-300">Target Customers</p>
              <p className="text-2xl font-bold text-white mt-1">
                {result.total_customers.toLocaleString()}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
              <p className="text-sm text-gray-300">Customers Retained</p>
              <p className="text-2xl font-bold text-green-400 mt-1">
                {result.customers_retained.toLocaleString()}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
              <p className="text-sm text-gray-300">Campaign Cost</p>
              <p className="text-2xl font-bold text-red-400 mt-1">
                ${result.campaign_cost.toLocaleString()}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
              <p className="text-sm text-gray-300">Expected Revenue</p>
              <p className="text-2xl font-bold text-green-400 mt-1">
                ${result.expected_revenue.toLocaleString()}
              </p>
            </div>
          </div>

          <div className={`rounded-lg p-6 ${
            result.net_roi > 0 ? 'bg-green-900 border-2 border-green-600' : 'bg-red-900 border-2 border-red-600'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-200">Net ROI</p>
                <p className={`text-4xl font-bold mt-1 ${
                  result.net_roi > 0 ? 'text-green-300' : 'text-red-300'
                }`}>
                  ${result.net_roi.toLocaleString()}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-200">ROI %</p>
                <p className={`text-4xl font-bold mt-1 ${
                  result.roi_percentage > 0 ? 'text-green-300' : 'text-red-300'
                }`}>
                  {result.roi_percentage.toFixed(0)}%
                </p>
              </div>
            </div>
          </div>

          {result.net_roi > 0 ? (
            <div className="bg-green-900 border border-green-600 rounded-lg p-4">
              <p className="text-sm text-green-200">
                ✅ <strong>Campaign is profitable!</strong> Expected to generate ${result.net_roi.toLocaleString()} in net revenue.
              </p>
            </div>
          ) : (
            <div className="bg-red-900 border border-red-600 rounded-lg p-4">
              <p className="text-sm text-red-200">
                ❌ <strong>Campaign may not be profitable.</strong> Consider adjusting cost or targeting different segments.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
