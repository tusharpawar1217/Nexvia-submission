'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import StatsCards from './components/StatsCards'
import SegmentsChart from './components/SegmentsChart'
import ROISimulator from './components/ROISimulator'
import HighRiskCustomers from './components/HighRiskCustomers'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Dashboard() {
  const [stats, setStats] = useState<any>(null)
  const [segments, setSegments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [statsRes, segmentsRes] = await Promise.all([
        axios.get(`${API_URL}/api/stats`),
        axios.get(`${API_URL}/api/segments`)
      ])
      setStats(statsRes.data)
      setSegments(segmentsRes.data)
    } catch (error) {
      console.error('Error fetching data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-300">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">
                Customer Churn ROI Dashboard
              </h1>
              <p className="mt-1 text-sm text-gray-300">
                ML-powered churn prediction and retention ROI analysis
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-900 text-green-300 border border-green-700">
                <span className="w-2 h-2 mr-2 bg-green-500 rounded-full animate-pulse"></span>
                Live
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <StatsCards stats={stats} />

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
          {/* Segments Chart */}
          <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Customer Segments
            </h2>
            <SegmentsChart segments={segments} />
          </div>

          {/* ROI Simulator */}
          <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              ROI Simulator
            </h2>
            <ROISimulator segments={segments} />
          </div>
        </div>

        {/* High Risk Customers Table */}
        <div className="mt-8">
          <HighRiskCustomers />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-400">
            Customer Churn ROI Prediction System • Built with FastAPI + Next.js
          </p>
        </div>
      </footer>
    </div>
  )
}
