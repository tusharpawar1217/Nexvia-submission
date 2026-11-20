'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function HighRiskCustomers() {
  const [customers, setCustomers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchHighRiskCustomers()
  }, [])

  const fetchHighRiskCustomers = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/high-risk?limit=50`)
      setCustomers(response.data)
    } catch (error) {
      console.error('Error fetching high-risk customers:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 p-6">
        <p className="text-gray-400 text-center">Loading high-risk customers...</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700">
      <div className="px-6 py-4 border-b border-gray-700">
        <h2 className="text-xl font-semibold text-white">
          High-Risk Customers
        </h2>
        <p className="text-sm text-gray-400 mt-1">
          Top 50 customers at risk of churn, sorted by CLV
        </p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-700">
          <thead className="bg-gray-750">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Customer ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Churn Risk
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                CLV
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Segment
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Tenure
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Monthly Charges
              </th>
            </tr>
          </thead>
          <tbody className="bg-gray-800 divide-y divide-gray-700">
            {customers.map((customer, index) => (
              <tr key={index} className="hover:bg-gray-750">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-200">
                  {customer.customer_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      customer.churn_probability > 0.8 ? 'bg-red-900 text-red-200 border border-red-700' :
                      customer.churn_probability > 0.6 ? 'bg-orange-900 text-orange-200 border border-orange-700' :
                      'bg-yellow-900 text-yellow-200 border border-yellow-700'
                    }`}>
                      {(customer.churn_probability * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-green-400 font-semibold">
                  ${customer.clv_predicted.toFixed(0)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  {customer.segment_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  {customer.tenure} mo
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  ${customer.monthly_charges.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
