import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import base64

class PortfolioVisualizer:
    """
    Creates interactive visualizations and a professional HTML dashboard for portfolio data.
    """
    
    def __init__(self):
        # Define a professional color palette
        self.colors = {
            'strategy': '#2563EB', # Royal Blue
            'benchmark': '#6B7280', # Gray
            'drawdown': '#EF4444', # Red
            'positive': '#10B981', # Green
            'negative': '#EF4444', # Red
            'bg': '#FFFFFF',
            'grid': '#F3F4F6'
        }

    def create_dashboard_html(self, df: pd.DataFrame, stats: dict, holdings_df: pd.DataFrame = None, trades_df: pd.DataFrame = None, stress_df: pd.DataFrame = None, mc_results: dict = None, vol_cone_data: dict = None) -> str:
        """Generates the full HTML dashboard."""
        
        # Create Charts
        fig_equity = self.plot_equity_curve(df)
        fig_drawdown = self.plot_drawdown(df)
        fig_monthly = self.plot_monthly_heatmap(df['Portfolio_Return'])
        fig_exposure = self.plot_exposure(df)
        fig_beta = self.plot_rolling_beta(df)
        
        fig_sharpe = self.plot_rolling_sharpe(df)
        fig_var = self.plot_var_history(df)
        
        # Monte Carlo Chart (New)
        plot_mc_div = ""
        if mc_results:
            fig_mc = self.plot_monte_carlo_cone(mc_results)
            plot_mc_div = fig_mc.to_html(full_html=False, include_plotlyjs=False)
            
        # Vol Cone Chart (New)
        plot_vol_cone_div = ""
        if vol_cone_data:
            fig_vol = self.plot_volatility_cone(vol_cone_data)
            plot_vol_cone_div = fig_vol.to_html(full_html=False, include_plotlyjs=False)
            
        # Attribution Chart (New)
        fig_attr = self.plot_attribution(df)
        plot_attr_div = fig_attr.to_html(full_html=False, include_plotlyjs=False)
        
        # Convert to HTML
        plot_equity_div = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
        plot_drawdown_div = fig_drawdown.to_html(full_html=False, include_plotlyjs=False)
        plot_monthly_div = fig_monthly.to_html(full_html=False, include_plotlyjs=False)
        plot_exposure_div = fig_exposure.to_html(full_html=False, include_plotlyjs=False)
        plot_beta_div = fig_beta.to_html(full_html=False, include_plotlyjs=False)
        plot_sharpe_div = fig_sharpe.to_html(full_html=False, include_plotlyjs=False)
        plot_var_div = fig_var.to_html(full_html=False, include_plotlyjs=False)
        
        # Generate Holdings Table HTML if available
        holdings_html = ""
        if holdings_df is not None and not holdings_df.empty:
            rows = []
            for _, row in holdings_df.iterrows():
                rows.append(f"""
                <tr>
                    <td><b>{row['Ticker']}</b></td>
                    <td class="pos-val">{row['Shares']:,.0f}</td>
                    <td class="pos-val">{row['Price']:,.2f}</td>
                    <td class="pos-val">{row['Market Value']:,.0f}</td>
                    <td class="pos-val" style="font-weight:bold; color:#2563EB">{row['Weight %']:.2f}%</td>
                </tr>
                """)
            
            holdings_html = f"""
            <div class="full-width table-container">
                <h3 style="margin-top:0; color:#374151">Current Holdings</h3>
                <table>
                    <thead>
                        <tr>
                            <th style="text-align: left;">Ticker</th>
                            <th>Shares</th>
                            <th>Price(USD)</th>
                            <th>Market Value(USD)</th>
                            <th>Weight %</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
            """
            
        # Generate Stress Test Table HTML
        stress_html = ""
        if stress_df is not None and not stress_df.empty:
            s_rows = []
            for _, row in stress_df.iterrows():
                pnl_color = "#EF4444" if row['Est. PnL Impact'] < 0 else "#10B981"
                s_rows.append(f"""
                <tr>
                    <td><b>{row['Scenario']}</b></td>
                    <td class="pos-val">{row['Market Shock']:.0%}</td>
                    <td class="pos-val">{row['Est. Portfolio Return']:.2%}</td>
                    <td class="pos-val" style="color:{pnl_color}; font-weight:bold;">{row['Est. PnL Impact']:,.0f}</td>
                </tr>
                """)
            
            stress_html = f"""
            <div class="full-width table-container" style="margin-bottom: 24px;">
                <h3 style="margin-top:0; color:#374151">Stress Test Analysis (Beta-Adj)</h3>
                <table>
                    <thead>
                        <tr>
                            <th style="text-align: left;">Scenario</th>
                            <th>Market Shock</th>
                            <th>Est. Return</th>
                            <th>Est. PnL Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(s_rows)}
                    </tbody>
                </table>
            </div>
            """

        # Generate Trade History HTML if available
        trades_html = ""
        if trades_df is not None and not trades_df.empty:
            t_rows = []
            # Sort by Date desc
            trades_sorted = trades_df.sort_values('Date', ascending=False)
            for _, row in trades_sorted.iterrows():
                direction = "BUY" if row['Shares'] > 0 else "SELL"
                color = "#10B981" if direction == "BUY" else "#EF4444"
                date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                
                t_rows.append(f"""
                <tr>
                    <td>{date_str}</td>
                    <td><b>{row['Ticker']}</b></td>
                    <td style="color:{color}; font-weight:bold;">{direction}</td>
                    <td class="pos-val">{abs(row['Shares']):,.0f}</td>
                    <td class="pos-val">{row.get('Price', 0.0):,.2f}</td>
                </tr>
                """)
                
            trades_html = f"""
            <div class="full-width table-container" style="margin-top: 24px;">
                <h3 style="margin-top:0; color:#374151">Trade History</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th style="text-align: left;">Ticker</th>
                            <th style="text-align: left;">Action</th>
                            <th>Shares</th>
                            <th>Price(USD)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(t_rows)}
                    </tbody>
                </table>
            </div>
            """

        # HTML Template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Analytics Report</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #F3F4F6; margin: 0; padding: 40px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                
                .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; }}
                .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; color: #111827; }}
                .header .meta {{ color: #6B7280; font-size: 14px; }}
                
                /* Metrics Grid */
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
                .metric-card .label {{ font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
                .metric-card .value {{ font-size: 24px; font-weight: 700; color: #111827; }}
                .metric-card .sub {{ font-size: 12px; color: #9CA3AF; margin-top: 4px; }}

                /* Charts Grid */
                .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
                .chart-card {{
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    min-height: 450px;
                }}
                .full-width {{ grid-column: 1 / -1; }}
                
                /* Table Styles */
                .table-container {{ background: white; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ text-align: right; padding: 12px; border-bottom: 2px solid #E5E7EB; color: #6B7280; font-size: 13px; font-weight: 600; text-transform: uppercase; }}
                th:first-child {{ text-align: left; }} /* Specific ticker align */
                td {{ padding: 12px; border-bottom: 1px solid #F3F4F6; color: #374151; font-size: 14px; }}
                tr:last-child td {{ border-bottom: none; }}
                .pos-val {{ font-family: 'Consolas', monospace; text-align: right; }}
                td:first-child {{ text-align: left; }}
                
                .footer {{ margin-top: 40px; text-align: center; color: #9CA3AF; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <h1>Portfolio Performance Report</h1>
                        <div class="meta">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Benchmark: {stats.get('Benchmark', 'N/A')}</div>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Total Return</div>
                        <div class="value" style="color: {'#10B981' if stats['Total_Return'] > 0 else '#EF4444'}">{stats['Total_Return']:.2%}</div>
                        <div class="sub">CAGR: {stats['CAGR']:.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value">{stats['Sharpe_Ratio']:.2f}</div>
                        <div class="sub">Sortino: {stats.get('Sortino_Ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Max Drawdown</div>
                        <div class="value" style="color: #EF4444">{stats['Max_Drawdown']:.2%}</div>
                        <div class="sub">Peak-to-Valley</div>
                    </div>
                     <div class="metric-card">
                        <div class="label">Alpha</div>
                        <div class="value">{stats['Alpha']:.2f}</div>
                        <div class="sub">Jensen's Alpha</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">End NAV</div>
                        <div class="value">${stats['End_NAV']:,.0f}</div>
                    </div>
                </div>

                <!-- Risk Analytics Section -->
                <h3 style="margin-bottom: 16px; color: #374151; font-weight: 600;">Risk Analytics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">VaR (95%)</div>
                        <div class="value" style="color: #F59E0B">{stats.get('VaR_95_Hist', 0):.2%}</div>
                        <div class="sub">Historical 1-Day</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">CVaR (95%)</div>
                        <div class="value" style="color: #EF4444">{stats.get('CVaR_95', 0):.2%}</div>
                        <div class="sub">Exp. Shortfall</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Ann. Volatility</div>
                        <div class="value">{stats['Volatility']:.2%}</div>
                        <div class="sub">Annualized</div>
                    </div>
                     <div class="metric-card">
                        <div class="label">Beta</div>
                        <div class="value">{stats['Beta']:.2f}</div>
                        <div class="sub">vs {stats.get('Benchmark', 'Benchmark')}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Turnover</div>
                        <div class="value">{stats.get('Turnover_Annualized', 0):.1f}x</div>
                        <div class="sub">Annualized</div>
                    </div>
                </div>

                <div class="charts-grid">
                    <div class="chart-card full-width">
                        {plot_equity_div}
                    </div>
                    
                    <div class="chart-card">
                        {plot_drawdown_div}
                    </div>
                    <div class="chart-card">
                        {plot_monthly_div}
                    </div>
                    
                     <div class="chart-card">
                        {plot_exposure_div}
                    </div>
                    <div class="chart-card">
                        {plot_beta_div}
                    </div>
                </div>

                <h3 style="margin-bottom: 16px; color: #374151; font-weight: 600;">Risk Insights</h3>
                <div class="charts-grid">
                    <div class="chart-card">
                        {plot_sharpe_div}
                    </div>
                    <div class="chart-card">
                        {plot_var_div}
                    </div>
                    <div class="chart-card">
                        {plot_vol_cone_div}
                    </div>
                    <div class="chart-card">
                         <!-- Placeholder or Monte Carlo if avail -->
                        {plot_mc_div}
                    </div>
                    
                    <div class="chart-card full-width">
                        {plot_attr_div}
                    </div>
                     
                    {holdings_html}
                    
                    {stress_html}
                    
                    {trades_html}
                </div>
                
                <div class="footer">
                    CONFIDENTIAL - Generated by Portfolio Analytics Engine
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

    def plot_exposure(self, df: pd.DataFrame) -> go.Figure:
        """Plots Gross and Net Leverage over time."""
        fig = go.Figure()
        
        # Explicit conversion
        x_data = df.index.tolist()
        
        # Gross Leverage
        if 'Gross_Leverage' in df.columns:
            y_gross = df['Gross_Leverage'].tolist()
            fig.add_trace(go.Scatter(
                x=x_data, y=y_gross, 
                name='Gross Lev',
                line=dict(color='#8B5CF6', width=2),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
        # Net Leverage
        if 'Net_Leverage' in df.columns:
            y_net = df['Net_Leverage'].tolist()
            fig.add_trace(go.Scatter(
                x=x_data, y=y_net, 
                name='Net Lev',
                line=dict(color='#10B981', width=2)
            ))
            
        fig.update_layout(
            title='Exposure Analysis (Leverage)',
            xaxis_title='',
            yaxis_title='Leverage Ratio',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis=dict(range=[-1, 2.5]) # Fixed range as requested
        )
        return fig

    def plot_equity_curve(self, df: pd.DataFrame) -> go.Figure:
        """Plots Portfolio vs Benchmark rebased to 100."""
        df = df.copy()
        
        current_eq_0 = df['Total_Equity'].iloc[0]
        ret_0 = df['Portfolio_Return'].iloc[0]
        denom = 1.0 + ret_0
        derived_start_equity = current_eq_0 / denom if denom != 0 else current_eq_0
        
        df['Strategy'] = (df['Total_Equity'] / derived_start_equity) * 100
        
        if 'Benchmark_Price' in df.columns and df['Benchmark_Price'].iloc[0] > 0:
             df['Benchmark'] = (df['Benchmark_Price'] / df['Benchmark_Price'].iloc[0]) * 100
        else:
             df['Benchmark'] = (1 + df['Benchmark_Return']).cumprod() * 100
        
        # Explicit conversion to list to avoid serialization issues
        x_data = df.index.tolist()
        y_strat = df['Strategy'].tolist()
        y_bench = df['Benchmark'].tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_strat, name='Strategy', line=dict(color=self.colors['strategy'], width=2.5)))
        fig.add_trace(go.Scatter(x=x_data, y=y_bench, name='Benchmark', line=dict(color=self.colors['benchmark'], dash='dot', width=1.5)))
        
        fig.update_layout(
            title='Cumulative Returns (Rebased to 100)',
            xaxis_title='',
            yaxis_title='Value',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_drawdown(self, df: pd.DataFrame) -> go.Figure:
        """Plots drawdown chart."""
        equity = df['Total_Equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        # Explicit conversion
        x_data = drawdown.index.tolist()
        y_data = drawdown.tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, 
            fill='tozeroy', 
            name='Drawdown', 
            line=dict(color=self.colors['drawdown'], width=1.5),
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        fig.update_layout(
            title='Drawdown Profile',
            yaxis_title='Drawdown %',
            yaxis_tickformat='.0%',
            template='plotly_white',
            margin=dict(l=40, r=40, t=60, b=80) # Increased bottom margin
        )
        return fig

    def plot_monthly_heatmap(self, returns_series: pd.Series) -> go.Figure:
        """Creates a monthly return heatmap."""
        monthly_ret = returns_series.resample('ME').apply(lambda x: (1+x).prod() - 1)
        
        monthly_ret = monthly_ret.to_frame(name='Return')
        monthly_ret['Year'] = monthly_ret.index.year
        monthly_ret['Month'] = monthly_ret.index.strftime('%b')
        monthly_ret['MonthNum'] = monthly_ret.index.month
        
        pivot = monthly_ret.pivot_table(index='Year', columns='MonthNum', values='Return')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot = pivot.reindex(columns=range(1, 13))
        
        # Explicit conversion for heatmap
        # Convert years to strings to force categorical axis (fixes rendering issue)
        y_years = [str(y) for y in pivot.index.tolist()]
        z_values = pivot.values.tolist()
        
        # Use graph_objects Heatmap for maximum control avoiding serialization issues
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=month_names,
            y=y_years,
            colorscale='RdYlGn',
            zmid=0,
            texttemplate="%{z:.1%}",
            textfont={"size": 10},
            ygap=2, # Add gap between cells
            xgap=2
        ))
        
        fig.update_layout(
            title="Monthly Returns",
            margin=dict(l=40, r=40, t=60, b=80), # Increased bottom margin
            yaxis=dict(type='category') # Enforce categorical axis
        )
        return fig

    def plot_rolling_beta(self, df: pd.DataFrame) -> go.Figure:
        """Plots 60-Day Rolling Beta."""
        fig = go.Figure()
        
        if 'Rolling_Beta' in df.columns:
            # Explicit conversion
            x_data = df.index.tolist()
            y_beta = df['Rolling_Beta'].tolist()
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_beta,
                name='Rolling Beta (60d)',
                line=dict(color='#F59E0B', width=2)
            ))
            
            # Add 1.0 reference line
            fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Market Risk (Beta=1)")
            
        fig.update_layout(
            title='Rolling 60-Day Beta to Benchmark',
            yaxis_title='Beta',
            template='plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_monte_carlo_cone(self, mc_results: dict) -> go.Figure:
        """Plots Monte Carlo Simulation (VaR Cone)."""
        paths = mc_results['paths'] # (n_sims, days)
        days = mc_results['days']
        
        # Calculate percentiles for each day
        p5 = np.percentile(paths, 5, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        
        # Date axis (Future)
        start_date = pd.Timestamp.now().normalize()
        dates = [start_date + pd.Timedelta(days=i) for i in range(days)]
        
        fig = go.Figure()
        
        # Upper Bound (95th)
        fig.add_trace(go.Scatter(
            x=dates, y=p95,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='95th Percentile'
        ))
        
        # Lower Bound (5th) with fill
        fig.add_trace(go.Scatter(
            x=dates, y=p5,
            mode='lines',
            line=dict(width=0),
            fill='tonexty', # Fill to previous trace (p95)
            fillcolor='rgba(37, 99, 235, 0.1)', # Light Blue
            name='90% Confidence Interval'
        ))
        
        # Median (50th)
        fig.add_trace(go.Scatter(
            x=dates, y=p50,
            mode='lines',
            line=dict(color=self.colors['strategy'], width=2),
            name='Median Projection'
        ))
        
        # Annotate VaR Floor
        final_var_level = p5[-1]
        fig.add_hline(y=final_var_level, line_dash="dash", line_color="#EF4444", 
                      annotation_text=f"VaR 95% Floor: ${final_var_level:,.0f}", 
                      annotation_position="bottom right")
        
        fig.update_layout(
            title=f'Monte Carlo Simulation (Projected {days} Days)',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        return fig

    def plot_rolling_sharpe(self, df: pd.DataFrame) -> go.Figure:
        """Plots 6-Month Rolling Sharpe Ratio."""
        fig = go.Figure()
        if 'Rolling_Sharpe' in df.columns:
            # Explicit conversion
            x_data = df.index.tolist()
            y_data = df['Rolling_Sharpe'].tolist()
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                name='Rolling Sharpe (6M)',
                line=dict(color='#8B5CF6', width=2), # Purple
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
            # Add 0 line
            fig.add_hline(y=0, line_color="gray", line_width=1)
            
        fig.update_layout(
            title='Rolling Sharpe Ratio (60 days)',
            yaxis_title='Sharpe',
            template='plotly_white',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_var_history(self, df: pd.DataFrame) -> go.Figure:
        """Plots Historical VaR Trend."""
        fig = go.Figure()
        if 'Rolling_VaR_95' in df.columns:
            x_data = df.index.tolist()
            y_data = (-df['Rolling_VaR_95']).tolist() # Plot as negative
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                name='VaR 95% (Rolling 1Y)',
                line=dict(color='#EF4444', width=2)
            ))
            
        fig.update_layout(
            title='Dynamic Risk: Rolling 1-Year VaR (95%)',
            yaxis_title='VaR (%)',
            template='plotly_white',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_volatility_cone(self, vol_data: dict) -> go.Figure:
        """Plots Volatility Cone (Box Plot of Realized Vol per Horizon)."""
        fig = go.Figure()
        
        horizons = []
        
        # Sort keys
        sorted_keys = sorted(vol_data.keys())
        
        for w in sorted_keys:
            series = vol_data[w].dropna()
            label = f"{w} Days"
            horizons.append(label)
            
            fig.add_trace(go.Box(
                y=series,
                name=label,
                boxpoints=False, # Don't show all points
                marker_color='#2563EB'
            ))
            
        fig.update_layout(
            title='Volatility Cone (Realized Vol Distribution)',
            yaxis_title='Annualized Volatility',
            xaxis_title='Time Horizon',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        return fig

    def plot_attribution(self, df: pd.DataFrame) -> go.Figure:
        """Plots Cumulative Attribution: Market Timing (Beta) vs Selection (Alpha)."""
        fig = go.Figure()
        
        if 'Cum_Selection' in df.columns:
            x_data = df.index.tolist()
            # We rebase them to start at 0% return (start value 100 -> 0)
            y_selection = (df['Cum_Selection'] - 100).tolist()
            y_systematic = (df['Cum_Systematic'] - 100).tolist()
            y_total = (df['Strategy'] - 100).tolist() if 'Strategy' in df.columns else []
            
            # 1. Total Return
            if y_total:
                 fig.add_trace(go.Scatter(
                    x=x_data, y=y_total,
                    name='Total Return',
                    line=dict(color='#111827', width=3) # Black
                ))
            
            # 2. Selection (Alpha)
            fig.add_trace(go.Scatter(
                x=x_data, y=y_selection,
                name='Stock Selection (Alpha)',
                line=dict(color='#10B981', width=2), # Green
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            # 3. Market (Beta)
            fig.add_trace(go.Scatter(
                x=x_data, y=y_systematic,
                name='Market Timing (Beta)',
                line=dict(color='#6B7280', width=2, dash='dot') # Gray dashed
            ))
            
        fig.update_layout(
            title='Performance Attribution (Cumulative Impact)',
            yaxis_title='Cumulative Return (%)',
            template='plotly_white',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
