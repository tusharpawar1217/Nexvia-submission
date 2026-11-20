"""
Customer Churn ROI Dashboard - Streamlit App
Interactive dashboard for exploring churn predictions, survival analysis, CLV, and ROI simulations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import execute_query
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn ROI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric { font-size: 2.5rem; font-weight: bold; }
    .metric-label { font-size: 1rem; color: #666; }
    .risk-high { color: #ff4444; }
    .risk-medium { color: #ffaa00; }
    .risk-low { color: #44ff44; }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(ttl=600)
def load_customer_data():
    """Load complete customer data with predictions."""
    query = """
        SELECT 
            c.customer_id,
            c.gender,
            c.senior_citizen,
            c.partner,
            c.dependents,
            c.tenure,
            c.phone_service,
            c.multiple_lines,
            c.internet_service,
            c.online_security,
            c.online_backup,
            c.device_protection,
            c.tech_support,
            c.streaming_tv,
            c.streaming_movies,
            c.contract,
            c.paperless_billing,
            c.payment_method,
            c.monthly_charges,
            c.total_charges,
            c.churn_value,
            p.churn_probability,
            p.expected_lifetime_months,
            p.median_survival_months,
            p.clv_predicted,
            p.clv_historical,
            p.segment_id,
            p.segment_name,
            f.service_count,
            f.has_streaming,
            f.has_protection,
            f.has_support,
            f.has_autopay
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        JOIN features f ON c.customer_id = f.customer_id
    """
    return execute_query(query)

@st.cache_data(ttl=600)
def load_segment_summary():
    """Load segment-level statistics."""
    query = """
        SELECT 
            segment_name,
            COUNT(*) as customer_count,
            ROUND(AVG(churn_probability)::numeric, 3) as avg_churn_prob,
            ROUND(AVG(clv_predicted)::numeric, 2) as avg_clv,
            ROUND(AVG(expected_lifetime_months)::numeric, 1) as avg_lifetime,
            ROUND(SUM(clv_predicted)::numeric, 2) as total_clv
        FROM predictions
        GROUP BY segment_name
        ORDER BY avg_clv DESC
    """
    return execute_query(query)

@st.cache_resource
def load_models():
    """Load trained models."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    return {
        'churn': joblib.load(os.path.join(models_dir, 'churn_model.pkl')),
        'shap': joblib.load(os.path.join(models_dir, 'shap_explainer.pkl')),
        'survival': joblib.load(os.path.join(models_dir, 'survival_model.pkl')),
        'segmentation': joblib.load(os.path.join(models_dir, 'segmentation_model.pkl'))
    }

# Sidebar navigation
st.sidebar.title("📊 Churn ROI Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Customer Explorer", "👥 Segments", "💰 ROI Simulator"]
)

# Load data
df = load_customer_data()
segments_df = load_segment_summary()

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.title("📊 Customer Churn Overview")
    st.markdown("---")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{len(df):,}",
            help="Total number of customers in database"
        )
    
    with col2:
        churn_rate = df['churn_value'].mean() * 100
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%",
            help="Historical churn rate"
        )
    
    with col3:
        high_risk = (df['churn_probability'] >= 0.5).sum()
        st.metric(
            "High-Risk Customers",
            f"{high_risk:,}",
            delta=f"{high_risk/len(df)*100:.1f}%",
            delta_color="inverse",
            help="Customers with churn probability >= 50%"
        )
    
    with col4:
        avg_clv = df['clv_predicted'].mean()
        st.metric(
            "Avg CLV",
            f"${avg_clv:,.0f}",
            help="Average predicted customer lifetime value"
        )
    
    st.markdown("---")
    
    # Revenue metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_value = df['clv_predicted'].sum()
        st.metric(
            "Total Customer Value",
            f"${total_value/1e6:.2f}M",
            help="Sum of all predicted CLV"
        )
    
    with col2:
        at_risk_value = df[df['churn_probability'] >= 0.5]['clv_predicted'].sum()
        st.metric(
            "Revenue at Risk",
            f"${at_risk_value/1e6:.2f}M",
            delta=f"{at_risk_value/total_value*100:.1f}%",
            delta_color="inverse",
            help="CLV of high-risk customers"
        )
    
    with col3:
        high_value_risk = df[(df['clv_predicted'] > df['clv_predicted'].quantile(0.75)) & 
                             (df['churn_probability'] > 0.5)]
        st.metric(
            "High-Value At-Risk",
            f"{len(high_value_risk):,}",
            help="High CLV customers with >50% churn risk - PRIORITY TARGETS"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Churn Risk Distribution")
        fig = px.histogram(
            df,
            x='churn_probability',
            nbins=50,
            title="Distribution of Churn Probabilities",
            labels={'churn_probability': 'Churn Probability', 'count': 'Number of Customers'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 CLV Distribution")
        fig = px.histogram(
            df,
            x='clv_predicted',
            nbins=50,
            title="Distribution of Customer Lifetime Value",
            labels={'clv_predicted': 'Predicted CLV ($)', 'count': 'Number of Customers'},
            color_discrete_sequence=['#2ca02c']
        )
        fig.add_vline(x=df['clv_predicted'].median(), line_dash="dash", 
                     line_color="orange", annotation_text="Median")
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment overview
    st.markdown("---")
    st.subheader("👥 Customer Segments Overview")
    
    # Segment metrics table
    segment_display = segments_df.copy()
    segment_display.columns = ['Segment', 'Count', 'Avg Churn Risk', 'Avg CLV', 'Avg Lifetime (mo)', 'Total CLV']
    segment_display['% of Total'] = (segment_display['Count'] / segment_display['Count'].sum() * 100).round(1)
    
    st.dataframe(
        segment_display.style.format({
            'Count': '{:,}',
            'Avg Churn Risk': '{:.1%}',
            'Avg CLV': '${:,.0f}',
            'Avg Lifetime (mo)': '{:.1f}',
            'Total CLV': '${:,.0f}',
            '% of Total': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Segment visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            segments_df,
            values='customer_count',
            names='segment_name',
            title="Customer Distribution by Segment",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segments_df,
            x='segment_name',
            y='total_clv',
            title="Total CLV by Segment",
            labels={'segment_name': 'Segment', 'total_clv': 'Total CLV ($)'},
            color='avg_churn_prob',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: CUSTOMER EXPLORER
# ============================================================================
elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Explorer")
    st.markdown("Search and analyze individual customers")
    st.markdown("---")
    
    # Search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_id = st.text_input(
            "Search Customer ID",
            placeholder="Enter customer ID (e.g., 1234-ABCDE)",
            help="Search for a specific customer"
        )
    
    with col2:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "High (≥50%)", "Medium (25-50%)", "Low (<25%)"]
        )
    
    with col3:
        segment_filter = st.selectbox(
            "Segment",
            ["All"] + df['segment_name'].unique().tolist()
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_id:
        filtered_df = filtered_df[filtered_df['customer_id'].str.contains(search_id, case=False, na=False)]
    
    if risk_filter == "High (≥50%)":
        filtered_df = filtered_df[filtered_df['churn_probability'] >= 0.5]
    elif risk_filter == "Medium (25-50%)":
        filtered_df = filtered_df[(filtered_df['churn_probability'] >= 0.25) & 
                                  (filtered_df['churn_probability'] < 0.5)]
    elif risk_filter == "Low (<25%)":
        filtered_df = filtered_df[filtered_df['churn_probability'] < 0.25]
    
    if segment_filter != "All":
        filtered_df = filtered_df[filtered_df['segment_name'] == segment_filter]
    
    st.info(f"**{len(filtered_df):,}** customers match your filters")
    
    # Display customer list
    if len(filtered_df) > 0:
        # Prepare display dataframe
        display_cols = [
            'customer_id', 'tenure', 'contract', 'monthly_charges',
            'churn_probability', 'expected_lifetime_months', 
            'clv_predicted', 'segment_name'
        ]
        
        display_df = filtered_df[display_cols].copy()
        display_df = display_df.sort_values('churn_probability', ascending=False)
        
        # Add risk label
        display_df['risk_label'] = pd.cut(
            display_df['churn_probability'],
            bins=[0, 0.25, 0.5, 1.0],
            labels=['🟢 Low', '🟡 Medium', '🔴 High']
        )
        
        st.dataframe(
            display_df.style.format({
                'churn_probability': '{:.1%}',
                'monthly_charges': '${:.2f}',
                'clv_predicted': '${:,.0f}',
                'expected_lifetime_months': '{:.1f}',
                'tenure': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Customer detail view
        st.markdown("---")
        st.subheader("📋 Customer Detail View")
        
        selected_id = st.selectbox(
            "Select a customer to view details",
            filtered_df['customer_id'].tolist(),
            index=0
        )
        
        if selected_id:
            customer = df[df['customer_id'] == selected_id].iloc[0]
            
            # Header
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_pct = customer['churn_probability'] * 100
                risk_color = "🔴" if risk_pct >= 50 else "🟡" if risk_pct >= 25 else "🟢"
                st.metric("Churn Risk", f"{risk_color} {risk_pct:.1f}%")
            
            with col2:
                st.metric("CLV", f"${customer['clv_predicted']:,.0f}")
            
            with col3:
                st.metric("Expected Lifetime", f"{customer['expected_lifetime_months']:.1f} mo")
            
            with col4:
                st.metric("Segment", customer['segment_name'])
            
            # Details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Account Information**")
                st.write(f"- **Tenure**: {customer['tenure']:.0f} months")
                st.write(f"- **Contract**: {customer['contract']}")
                st.write(f"- **Monthly Charges**: ${customer['monthly_charges']:.2f}")
                st.write(f"- **Total Charges**: ${customer['total_charges']:,.2f}")
                st.write(f"- **Payment Method**: {customer['payment_method']}")
                st.write(f"- **Autopay**: {'✅ Yes' if customer['has_autopay'] else '❌ No'}")
            
            with col2:
                st.markdown("**📦 Services**")
                st.write(f"- **Total Services**: {customer['service_count']:.0f}")
                st.write(f"- **Internet**: {customer['internet_service']}")
                st.write(f"- **Phone**: {'✅' if customer['phone_service'] == 'Yes' else '❌'}")
                st.write(f"- **Streaming**: {'✅' if customer['has_streaming'] else '❌'}")
                st.write(f"- **Protection**: {'✅' if customer['has_protection'] else '❌'}")
                st.write(f"- **Tech Support**: {'✅' if customer['has_support'] else '❌'}")
            
            # Survival curve (simplified visualization)
            st.markdown("---")
            st.subheader("📈 Predicted Survival Curve")
            
            # Create survival curve data
            months = np.arange(0, 73)
            baseline_hazard = 0.015  # Approximate monthly hazard
            survival_prob = np.exp(-baseline_hazard * months * (2 if customer['churn_probability'] > 0.5 else 1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=survival_prob,
                mode='lines',
                name='Survival Probability',
                line=dict(color='blue', width=2)
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="50% Survival")
            fig.add_vline(x=customer['expected_lifetime_months'], line_dash="dash",
                         line_color="green", annotation_text="Expected Lifetime")
            fig.update_layout(
                xaxis_title="Months",
                yaxis_title="Survival Probability",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if customer['churn_probability'] >= 0.5:
                if customer['clv_predicted'] > df['clv_predicted'].quantile(0.75):
                    st.error("🚨 **HIGH PRIORITY**: High-value customer at severe risk!")
                    st.markdown("""
                    **Recommended Actions:**
                    - Personal outreach from account manager
                    - Offer contract upgrade with discount
                    - Service bundle optimization
                    - Immediate satisfaction survey
                    """)
                else:
                    st.warning("⚠️ **MEDIUM PRIORITY**: Customer at risk of churning")
                    st.markdown("""
                    **Recommended Actions:**
                    - Targeted retention offer
                    - Service education and optimization
                    - Contract incentive if month-to-month
                    """)
            else:
                st.success("✅ **LOW RISK**: Customer is likely to stay")
                st.markdown("""
                **Recommended Actions:**
                - Upsell additional services
                - Encourage long-term contract
                - Maintain satisfaction through regular check-ins
                """)

# ============================================================================
# PAGE 3: SEGMENTS
# ============================================================================
elif page == "👥 Segments":
    st.title("👥 Customer Segments Analysis")
    st.markdown("---")
    
    # Segment selector
    selected_segment = st.selectbox(
        "Select a segment to analyze",
        ["All Segments"] + df['segment_name'].unique().tolist()
    )
    
    if selected_segment == "All Segments":
        segment_df = df
    else:
        segment_df = df[df['segment_name'] == selected_segment]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", f"{len(segment_df):,}")
    
    with col2:
        st.metric("Avg Churn Risk", f"{segment_df['churn_probability'].mean():.1%}")
    
    with col3:
        st.metric("Avg CLV", f"${segment_df['clv_predicted'].mean():,.0f}")
    
    with col4:
        st.metric("Total Value", f"${segment_df['clv_predicted'].sum()/1e6:.2f}M")
    
    st.markdown("---")
    
    # Segment comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contract Distribution")
        contract_dist = segment_df.groupby('contract').size().reset_index(name='count')
        fig = px.bar(
            contract_dist,
            x='contract',
            y='count',
            title=f"Contract Types - {selected_segment}",
            color='contract'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Service Adoption")
        service_metrics = pd.DataFrame({
            'Service': ['Phone', 'Streaming', 'Protection', 'Support', 'Autopay'],
            'Adoption %': [
                (segment_df['phone_service'] == 'Yes').mean() * 100,
                segment_df['has_streaming'].mean() * 100,
                segment_df['has_protection'].mean() * 100,
                segment_df['has_support'].mean() * 100,
                segment_df['has_autopay'].mean() * 100
            ]
        })
        fig = px.bar(
            service_metrics,
            x='Service',
            y='Adoption %',
            title=f"Service Adoption Rates - {selected_segment}",
            color='Adoption %',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tenure vs CLV scatter
    st.markdown("---")
    st.subheader("📊 Tenure vs CLV Analysis")
    
    fig = px.scatter(
        segment_df,
        x='tenure',
        y='clv_predicted',
        color='churn_probability',
        size='monthly_charges',
        hover_data=['customer_id', 'contract', 'segment_name'],
        title=f"Tenure vs CLV - {selected_segment}",
        labels={
            'tenure': 'Tenure (months)',
            'clv_predicted': 'Predicted CLV ($)',
            'churn_probability': 'Churn Risk'
        },
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment comparison table
    if selected_segment == "All Segments":
        st.markdown("---")
        st.subheader("📋 Segment Comparison")
        
        comparison = df.groupby('segment_name').agg({
            'customer_id': 'count',
            'churn_probability': 'mean',
            'clv_predicted': ['mean', 'sum'],
            'tenure': 'mean',
            'monthly_charges': 'mean',
            'service_count': 'mean',
            'has_autopay': 'mean'
        }).round(2)
        
        comparison.columns = ['Count', 'Avg Churn Risk', 'Avg CLV', 'Total CLV', 
                            'Avg Tenure', 'Avg Monthly', 'Avg Services', 'Autopay %']
        
        st.dataframe(
            comparison.style.format({
                'Count': '{:,}',
                'Avg Churn Risk': '{:.1%}',
                'Avg CLV': '${:,.0f}',
                'Total CLV': '${:,.0f}',
                'Avg Tenure': '{:.1f}',
                'Avg Monthly': '${:.2f}',
                'Avg Services': '{:.1f}',
                'Autopay %': '{:.1%}'
            }),
            use_container_width=True
        )

# ============================================================================
# PAGE 4: ROI SIMULATOR
# ============================================================================
elif page == "💰 ROI Simulator":
    st.title("💰 Retention Campaign ROI Simulator")
    st.markdown("Simulate the economics of customer retention campaigns")
    st.markdown("---")
    
    # Campaign parameters
    st.subheader("⚙️ Campaign Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        offer_cost = st.slider(
            "Retention Offer Cost ($)",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Cost per customer for retention offer (e.g., discount, gift)"
        )
        
        acceptance_rate = st.slider(
            "Acceptance Rate (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="% of targeted customers who accept the offer"
        ) / 100
        
        effectiveness = st.slider(
            "Retention Effectiveness (%)",
            min_value=10,
            max_value=80,
            value=30,
            step=5,
            help="% reduction in churn for customers who accept"
        ) / 100
    
    with col2:
        target_strategy = st.radio(
            "Targeting Strategy",
            [
                "All High-Risk (≥50%)",
                "Top 25% by CLV + High Risk",
                "Medium Risk (25-50%) Only",
                "Custom Risk Range"
            ]
        )
        
        if target_strategy == "Custom Risk Range":
            risk_min = st.slider("Min Churn Risk (%)", 0, 100, 40, 5) / 100
            risk_max = st.slider("Max Churn Risk (%)", 0, 100, 70, 5) / 100
        
        clv_multiplier = st.number_input(
            "CLV Realization Factor",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="% of predicted CLV actually realized (conservative: 0.8)"
        )
    
    # Apply targeting
    if target_strategy == "All High-Risk (≥50%)":
        targeted = df[df['churn_probability'] >= 0.5]
    elif target_strategy == "Top 25% by CLV + High Risk":
        clv_threshold = df['clv_predicted'].quantile(0.75)
        targeted = df[(df['clv_predicted'] >= clv_threshold) & (df['churn_probability'] >= 0.5)]
    elif target_strategy == "Medium Risk (25-50%) Only":
        targeted = df[(df['churn_probability'] >= 0.25) & (df['churn_probability'] < 0.5)]
    else:  # Custom
        targeted = df[(df['churn_probability'] >= risk_min) & (df['churn_probability'] <= risk_max)]
    
    # Calculate ROI
    st.markdown("---")
    st.subheader("📊 Campaign Results")
    
    n_targeted = len(targeted)
    n_accept = int(n_targeted * acceptance_rate)
    n_retained = int(n_accept * effectiveness)
    
    avg_clv = targeted['clv_predicted'].mean()
    total_clv_targeted = targeted['clv_predicted'].sum()
    recovered_clv = n_retained * avg_clv * clv_multiplier
    
    campaign_cost = n_targeted * offer_cost
    net_benefit = recovered_clv - campaign_cost
    roi = (net_benefit / campaign_cost * 100) if campaign_cost > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Customers Targeted",
            f"{n_targeted:,}",
            help="Number of customers matching targeting criteria"
        )
    
    with col2:
        st.metric(
            "Customers Retained",
            f"{n_retained:,}",
            delta=f"{n_retained/n_targeted*100:.1f}%" if n_targeted > 0 else "0%",
            help="Customers who accept offer and are successfully retained"
        )
    
    with col3:
        st.metric(
            "Campaign Cost",
            f"${campaign_cost:,.0f}",
            help="Total cost of retention offers"
        )
    
    with col4:
        roi_delta = "normal" if roi > 0 else "inverse"
        st.metric(
            "ROI",
            f"{roi:.1f}%",
            delta="Positive" if roi > 0 else "Negative",
            delta_color=roi_delta,
            help="Return on Investment: (Recovered CLV - Cost) / Cost"
        )
    
    # Financial breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Recovered CLV",
            f"${recovered_clv:,.0f}",
            help="Total CLV recovered from retained customers"
        )
    
    with col2:
        st.metric(
            "Net Benefit",
            f"${net_benefit:,.0f}",
            delta="Profit" if net_benefit > 0 else "Loss",
            delta_color="normal" if net_benefit > 0 else "inverse",
            help="Recovered CLV minus campaign cost"
        )
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Campaign funnel
        funnel_data = pd.DataFrame({
            'Stage': ['Targeted', 'Accept Offer', 'Retained'],
            'Count': [n_targeted, n_accept, n_retained]
        })
        
        fig = px.funnel(
            funnel_data,
            x='Count',
            y='Stage',
            title="Campaign Funnel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Financial breakdown
        financial_data = pd.DataFrame({
            'Category': ['Recovered CLV', 'Campaign Cost', 'Net Benefit'],
            'Amount': [recovered_clv, -campaign_cost, net_benefit],
            'Type': ['Revenue', 'Cost', 'Profit' if net_benefit > 0 else 'Loss']
        })
        
        fig = px.bar(
            financial_data,
            x='Category',
            y='Amount',
            title="Financial Breakdown",
            color='Type',
            color_discrete_map={'Revenue': 'green', 'Cost': 'red', 'Profit': 'blue', 'Loss': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sensitivity analysis
    st.markdown("---")
    st.subheader("📈 Sensitivity Analysis")
    
    # Vary effectiveness
    effectiveness_range = np.arange(0.1, 0.81, 0.05)
    roi_by_effectiveness = []
    
    for eff in effectiveness_range:
        n_ret_temp = int(n_accept * eff)
        recovered_temp = n_ret_temp * avg_clv * clv_multiplier
        net_temp = recovered_temp - campaign_cost
        roi_temp = (net_temp / campaign_cost * 100) if campaign_cost > 0 else 0
        roi_by_effectiveness.append(roi_temp)
    
    sensitivity_df = pd.DataFrame({
        'Retention Effectiveness (%)': effectiveness_range * 100,
        'ROI (%)': roi_by_effectiveness
    })
    
    fig = px.line(
        sensitivity_df,
        x='Retention Effectiveness (%)',
        y='ROI (%)',
        title="ROI vs Retention Effectiveness",
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if roi > 50:
        st.success("✅ **EXCELLENT ROI** - This campaign is highly profitable! Proceed with confidence.")
    elif roi > 0:
        st.info("✅ **POSITIVE ROI** - Campaign is profitable but could be optimized.")
        st.markdown("""
        **Optimization Ideas:**
        - Refine targeting to focus on highest-value customers
        - Test lower offer costs while maintaining acceptance rates
        - Improve retention effectiveness through personalization
        """)
    else:
        st.error("❌ **NEGATIVE ROI** - Campaign would lose money with current parameters.")
        st.markdown("""
        **Recommended Changes:**
        - Reduce offer cost (target: <$150 per customer)
        - Focus on higher CLV customers only
        - Improve retention tactics to increase effectiveness
        - Consider alternative retention strategies
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Customer Churn ROI Dashboard**

Built with Streamlit, PostgreSQL, LightGBM, and Lifelines.

Data: 7,043 telecom customers
Models: Churn, Survival, CLV, Segmentation
""")
