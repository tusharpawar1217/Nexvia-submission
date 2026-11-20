"""
Plotly Dash Dashboard for Retail Demand Forecasting

Features:
- Hierarchy browser
- Forecast visualization with prediction intervals
- Model comparison
- Inventory optimization
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import requests
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import config

# API base URL
API_BASE = config.get('dashboard.api_base_url', 'http://127.0.0.1:8000')

# Create Dash app
app = dash.Dash(
    __name__,
    title=config.get('dashboard.title', 'Demand Forecasting Dashboard')
)

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("🛒 Retail Demand Forecasting Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        
        # Info banner
        html.Div([
            html.P([
                "📊 System Status: ",
                html.Span("21,343 total series | ", style={'fontWeight': 'bold'}),
                html.Span("2,134 series with trained models (10% sample)", style={'fontWeight': 'bold', 'color': '#e67e22'}),
                " | Dropdown shows only series with forecasts"
            ], style={'margin': 0, 'fontSize': 14})
        ], style={'backgroundColor': '#d1ecf1', 'padding': 10, 'borderRadius': 5, 'marginBottom': 20, 'textAlign': 'center'}),
        
        # Filters
        html.Div([
            html.Div([
                html.Label("State:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='state-filter',
                    options=[
                        {'label': 'California (CA)', 'value': 'CA'},
                        {'label': 'Texas (TX)', 'value': 'TX'},
                        {'label': 'Wisconsin (WI)', 'value': 'WI'},
                    ],
                    value='CA',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("Category:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='category-filter',
                    options=[
                        {'label': 'Foods', 'value': 'FOODS'},
                        {'label': 'Household', 'value': 'HOUSEHOLD'},
                        {'label': 'Hobbies', 'value': 'HOBBIES'},
                    ],
                    value='HOBBIES',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("Series:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='series-dropdown',
                    options=[],
                    value=None,
                    style={'width': '400px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Button('Load Forecast', id='load-button', n_clicks=0,
                       style={'padding': '10px 20px', 'fontSize': 16, 'cursor': 'pointer'})
        ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 5, 'marginBottom': 20}),
        
        # Status
        html.Div(id='status-message', style={'padding': 10, 'marginBottom': 20}),
        
        # Tabs
        dcc.Tabs([
            # Tab 1: Forecast
            dcc.Tab(label='📈 Forecast', children=[
                html.Div([
                    dcc.Graph(id='forecast-plot', style={'height': '500px'}),
                    
                    html.Div([
                        html.H3("Forecast Summary"),
                        html.Div(id='forecast-summary', style={'fontSize': 16})
                    ], style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 5, 'marginTop': 20})
                ], style={'padding': 20})
            ]),
            
            # Tab 2: Model Comparison
            dcc.Tab(label='📊 Model Comparison', children=[
                html.Div([
                    dcc.Graph(id='model-comparison', style={'height': '400px'}),
                    html.Div(id='metrics-table', style={'marginTop': 20})
                ], style={'padding': 20})
            ]),
            
            # Tab 3: Inventory Optimization
            dcc.Tab(label='📦 Inventory Optimization', children=[
                html.Div([
                    html.H3("Optimal Order Quantity"),
                    
                    html.Div([
                        html.Label("Service Level:", style={'fontWeight': 'bold', 'marginRight': 10}),
                        dcc.Slider(
                            id='service-level-slider',
                            min=0.80,
                            max=0.99,
                            step=0.01,
                            value=0.95,
                            marks={0.80: '80%', 0.85: '85%', 0.90: '90%', 0.95: '95%', 0.99: '99%'},
                        )
                    ], style={'marginBottom': 20}),
                    
                    html.Button('Calculate Order Quantity', id='optimize-button', n_clicks=0,
                               style={'padding': '10px 20px', 'fontSize': 16, 'cursor': 'pointer'}),
                    
                    html.Div(id='optimization-result', style={'marginTop': 20})
                ], style={'padding': 20})
            ]),
        ])
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': 20})
])


@app.callback(
    Output('series-dropdown', 'options'),
    [Input('state-filter', 'value'),
     Input('category-filter', 'value')]
)
def update_series_options(state, category):
    """Update series dropdown based on filters - ONLY show series with forecasts."""
    try:
        # Load forecast series IDs (only these have trained models)
        import pandas as pd
        from pathlib import Path
        
        forecasts_path = Path(__file__).parent.parent / 'models' / 'lgbm_forecasts.parquet'
        forecasts_df = pd.read_parquet(forecasts_path)
        available_series = forecasts_df['id'].unique()
        
        # Get hierarchy info for available series
        response = requests.get(
            f"{API_BASE}/hierarchy",
            params={'state_id': state, 'cat_id': category, 'limit': 500}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter to only series with forecasts
            options = [
                {'label': f"{s['id']} ({s['item_id']} at {s['store_id']})", 'value': s['id']}
                for s in data['series']
                if s['id'] in available_series
            ]
            
            if not options:
                # If no match, show any available series from the state
                state_series = [sid for sid in available_series if state in sid][:20]
                options = [
                    {'label': sid, 'value': sid}
                    for sid in state_series
                ]
            
            return options
        else:
            return []
    except Exception as e:
        print(f"Error loading series: {e}")
        # Fallback: show some available series
        try:
            forecasts_df = pd.read_parquet('models/lgbm_forecasts.parquet')
            sample_ids = forecasts_df['id'].unique()[:20]
            return [{'label': sid, 'value': sid} for sid in sample_ids]
        except:
            return []


@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('forecast-summary', 'children'),
     Output('status-message', 'children')],
    [Input('load-button', 'n_clicks')],
    [State('series-dropdown', 'value')]
)
def load_forecast(n_clicks, series_id):
    """Load and display forecast."""
    if n_clicks == 0 or not series_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select a series and click 'Load Forecast'")
        return empty_fig, "", ""
    
    try:
        # Get forecast
        response = requests.get(f"{API_BASE}/forecast/{series_id}", timeout=10)
        
        if response.status_code != 200:
            error_fig = go.Figure()
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = f"HTTP {response.status_code}: {response.text[:100]}"
            error_fig.update_layout(title=f"Error: {error_detail}")
            error_msg = html.Div([
                html.P(f"❌ Error loading forecast for {series_id}", style={'color': 'red', 'fontWeight': 'bold'}),
                html.P(f"Status: {response.status_code}", style={'color': 'red'}),
                html.P(f"Detail: {error_detail}", style={'color': 'red', 'fontSize': 12})
            ])
            return error_fig, "", error_msg
        
        # Parse JSON response
        try:
            data = response.json()
        except Exception as json_err:
            error_fig = go.Figure()
            error_fig.update_layout(title=f"JSON Parse Error")
            error_msg = html.Div([
                html.P(f"❌ Invalid JSON response", style={'color': 'red', 'fontWeight': 'bold'}),
                html.P(f"Error: {str(json_err)}", style={'color': 'red', 'fontSize': 12}),
                html.P(f"Response: {response.text[:200]}", style={'color': 'gray', 'fontSize': 10})
            ])
            return error_fig, "", error_msg
        
        # Convert None to 0 for plotting (NaN handling)
        import numpy as np
        
        def clean_data(values):
            """Replace None with 0 for plotting."""
            return [0 if v is None else v for v in values]
        
        actual = clean_data(data['actual'])
        forecast = clean_data(data['forecast'])
        q05 = clean_data(data['q05'])
        q25 = clean_data(data['q25'])
        q75 = clean_data(data['q75'])
        q95 = clean_data(data['q95'])
        
        # Create figure
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(actual))),
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Forecast (median)
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast))),
            y=forecast,
            mode='lines',
            name='Forecast (Median)',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # 90% Prediction interval
        fig.add_trace(go.Scatter(
            x=list(range(len(q95))),
            y=q95,
            mode='lines',
            name='Q95',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(q05))),
            y=q05,
            mode='lines',
            name='90% Prediction Interval',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Forecast for {series_id}",
            xaxis_title="Day",
            yaxis_title="Sales",
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            height=500
        )
        
        # Summary (using cleaned data)
        total_actual = sum(actual)
        total_forecast = sum(forecast)
        error = abs(total_actual - total_forecast)
        error_pct = (error / (total_actual + 1e-10)) * 100
        
        summary = html.Div([
            html.P(f"📅 Forecast Period: {data['forecast_days']} days"),
            html.P(f"📊 Total Actual Sales: {total_actual:.0f}"),
            html.P(f"🎯 Total Forecast: {total_forecast:.0f}"),
            html.P(f"📉 Error: {error:.0f} ({error_pct:.1f}%)"),
        ])
        
        status = html.Div(f"✅ Forecast loaded for {series_id}", style={'color': 'green'})
        
        return fig, summary, status
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, "", html.Div(f"❌ Error: {str(e)}", style={'color': 'red'})


@app.callback(
    [Output('model-comparison', 'figure'),
     Output('metrics-table', 'children')],
    [Input('load-button', 'n_clicks')]
)
def compare_models(n_clicks):
    """Compare model performance."""
    if n_clicks == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Load a forecast first")
        return empty_fig, ""
    
    try:
        # Get metrics for all models
        models = ['lgbm', 'naive', 'snaive', 'ma']
        metrics_data = []
        
        for model in models:
            try:
                response = requests.get(f"{API_BASE}/metrics", params={'model': model})
                if response.status_code == 200:
                    metrics_data.append(response.json())
            except:
                pass
        
        if not metrics_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No metrics available")
            return empty_fig, ""
        
        # Create bar chart
        models_list = [m['model'].upper() for m in metrics_data]
        wape_values = [m['wape'] for m in metrics_data]
        
        fig = go.Figure([
            go.Bar(x=models_list, y=wape_values, text=[f"{v:.3f}" for v in wape_values],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title="Model Comparison (WAPE)",
            xaxis_title="Model",
            yaxis_title="WAPE (lower is better)",
            height=400
        )
        
        # Create metrics table
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Model"),
                html.Th("MAE"),
                html.Th("RMSE"),
                html.Th("WAPE"),
                html.Th("Coverage 90%")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(m['model'].upper()),
                    html.Td(f"{m['mae']:.4f}"),
                    html.Td(f"{m['rmse']:.4f}"),
                    html.Td(f"{m['wape']:.4f}"),
                    html.Td(f"{m.get('coverage_90', 'N/A'):.2%}" if isinstance(m.get('coverage_90'), float) else "N/A")
                ]) for m in metrics_data
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': 20})
        
        return fig, table
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, ""


@app.callback(
    Output('optimization-result', 'children'),
    [Input('optimize-button', 'n_clicks')],
    [State('series-dropdown', 'value'),
     State('service-level-slider', 'value')]
)
def optimize_inventory(n_clicks, series_id, service_level):
    """Calculate optimal order quantity."""
    if n_clicks == 0 or not series_id:
        return ""
    
    try:
        response = requests.post(
            f"{API_BASE}/optimize",
            params={'series_id': series_id, 'service_level': service_level}
        )
        
        if response.status_code != 200:
            return html.Div(f"❌ Error: {response.json().get('detail', 'Unknown error')}", 
                          style={'color': 'red'})
        
        data = response.json()
        rec = data['recommendation']
        costs = data['costs']
        
        return html.Div([
            html.H4(f"Series: {data['series_id']} ({data['category']})"),
            html.Div([
                html.H5("📦 Recommendation:"),
                html.P(f"Order Quantity: {rec['order_qty']} units"),
                html.P(f"Expected Demand: {rec['expected_demand']:.1f} units"),
                html.P(f"Safety Stock: {rec['safety_stock']:.1f} units"),
                html.P(f"Service Level: {rec['service_level']:.0%}"),
            ], style={'backgroundColor': '#d4edda', 'padding': 15, 'borderRadius': 5, 'marginBottom': 20}),
            
            html.Div([
                html.H5("💰 Costs (ASSUMED):"),
                html.P(f"Unit Cost: ${costs['unit_cost']:.2f}"),
                html.P(f"Total Order Cost: ${costs['order_cost']:.2f}"),
                html.P(f"Holding Cost (28 days): ${costs['holding_cost_28d']:.2f}"),
                html.P(f"Stockout Penalty: ${costs['stockout_penalty']:.2f}"),
                html.P(costs['note'], style={'fontStyle': 'italic', 'color': '#856404'})
            ], style={'backgroundColor': '#fff3cd', 'padding': 15, 'borderRadius': 5})
        ])
        
    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}", style={'color': 'red'})


if __name__ == '__main__':
    app.run_server(
        host=config.get('dashboard.host', '127.0.0.1'),
        port=config.get('dashboard.port', 8050),
        debug=config.get('dashboard.debug', False)
    )
