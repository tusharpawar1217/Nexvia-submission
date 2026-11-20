'use client'

interface StatsCardsProps {
  stats: {
    total_customers: number
    avg_churn_probability: number
    avg_clv: number
    high_risk_customers: number
  } | null
}

export default function StatsCards({ stats }: StatsCardsProps) {
  if (!stats) return null

  const cards = [
    {
      title: 'Total Customers',
      value: stats.total_customers.toLocaleString(),
      subtitle: 'Active customers',
      color: 'blue',
      icon: '👥'
    },
    {
      title: 'Avg Churn Risk',
      value: `${(stats.avg_churn_probability * 100).toFixed(1)}%`,
      subtitle: 'Overall churn probability',
      color: 'yellow',
      icon: '⚠️'
    },
    {
      title: 'Avg CLV',
      value: `$${stats.avg_clv.toFixed(0)}`,
      subtitle: 'Per customer',
      color: 'green',
      icon: '💰'
    },
    {
      title: 'High Risk',
      value: stats.high_risk_customers.toLocaleString(),
      subtitle: 'Customers at risk',
      color: 'red',
      icon: '🚨'
    }
  ]

  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-900',
    yellow: 'bg-yellow-50 border-yellow-200 text-yellow-900',
    green: 'bg-green-50 border-green-200 text-green-900',
    red: 'bg-red-50 border-red-200 text-red-900'
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, index) => (
        <div
          key={index}
          className={`${colorClasses[card.color as keyof typeof colorClasses]} rounded-lg border p-6 shadow-sm hover:shadow-md transition-shadow`}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium opacity-75">{card.title}</p>
              <p className="mt-2 text-3xl font-bold">{card.value}</p>
              <p className="mt-1 text-xs opacity-60">{card.subtitle}</p>
            </div>
            <div className="text-4xl">{card.icon}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
