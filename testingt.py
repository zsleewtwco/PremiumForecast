import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Aging Factor and Trend Calculation ---
def get_aging_factor(age):
    aging_factors = {
        1: 1.0276, 2: 1.0560, 3: 1.0851, 4: 1.1151, 5: 1.1459, 6: 1.1775,
        7: 1.2100, 8: 1.2434, 9: 1.2778, 10: 1.3131, 11: 1.3493, 12: 1.3866,
        13: 1.4248, 14: 1.4642, 15: 1.5046, 16: 1.5461, 17: 1.5888, 18: 1.6327,
        19: 1.6778, 20: 1.7241, 21: 1.8025, 22: 1.8845, 23: 1.9701, 24: 2.0597,
        25: 2.1534, 26: 2.2513, 27: 2.3537, 28: 2.4607, 29: 2.5726, 30: 2.6895,
        31: 2.7707, 32: 2.8543, 33: 2.9404, 34: 3.0291, 35: 3.1205, 36: 3.2147,
        37: 3.3117, 38: 3.4116, 39: 3.5145, 40: 3.6206, 41: 3.7764, 42: 3.9388,
        43: 4.1083, 44: 4.2850, 45: 4.4694, 46: 4.6616, 47: 4.8622, 48: 5.0713,
        49: 5.2895, 50: 5.5171, 51: 5.6528, 52: 5.7918, 53: 5.9342, 54: 6.0801,
        55: 6.2296, 56: 6.3828, 57: 6.5398, 58: 6.7006, 59: 6.8654, 60: 7.0342,
        61: 7.1413, 62: 7.2499, 63: 7.3603, 64: 7.4723, 65: 7.5860, 66: 7.7127,
        67: 7.8415, 68: 7.9725, 69: 8.1057, 70: 8.2411, 71: 8.5190, 72: 8.8062,
        73: 9.1032, 74: 9.8006, 75: 10.5514, 76: 10.6875, 77: 10.8255,
        78: 10.9652, 79: 11.2544, 80: 11.5513, 81: 12.1205, 82: 12.7177,
        83: 13.3444, 84: 13.6512, 85: 13.9651, 86: 14.0337, 87: 14.1027,
        88: 14.1720
    }
    
    # Handle exact integers
    if age % 1 == 0:
        factor = aging_factors.get(int(age))
        if factor is not None:
            return factor
    
    # Handle exact half-integers (e.g., 42.5)
    elif age % 1 == 0.5:
        lower = int(age)
        upper = lower + 1
        if lower in aging_factors and upper in aging_factors:
            return round((aging_factors[lower] + aging_factors[upper]) / 2, 8)
    
    # Handle any other decimal values by interpolating between the two nearest integers
    lower = int(age)
    upper = lower + 1
    if lower in aging_factors and upper in aging_factors:
        # Linear interpolation
        fraction = age - lower
        interpolated = aging_factors[lower] + fraction * (aging_factors[upper] - aging_factors[lower])
        return round(interpolated, 8)
    
    # If age is outside the range, return the closest available value
    if age < min(aging_factors.keys()):
        return aging_factors[min(aging_factors.keys())]
    elif age > max(aging_factors.keys()):
        return aging_factors[max(aging_factors.keys())]
    
    # Fallback - should not reach here
    return 1.0

def calculate_member_risk_trend(current_age, projected_age):
    base = get_aging_factor(current_age)
    target = get_aging_factor(projected_age)
    if base is None or target is None or base == 0:
        return 1.0  # Default to no trend if calculation fails
    return round(target / base, 8)

# --- Stochastic Modeling Functions ---
def fit_claims_distribution(historical_claims):
    """Fit probability distribution to historical claims"""
    
    # Try multiple distributions
    distributions = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min
    }
    
    best_fit = None
    best_aic = np.inf
    
    for name, distribution in distributions.items():
        try:
            # Fit distribution
            params = distribution.fit(historical_claims)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(distribution.logpdf(historical_claims, *params))
            
            # Calculate AIC (Akaike Information Criterion)
            aic = 2 * len(params) - 2 * log_likelihood
            
            if aic < best_aic:
                best_aic = aic
                best_fit = (name, distribution, params)
                
        except:
            continue
    
    return best_fit

def stochastic_projection(df, projection_years, n_simulations=10000):
    """Run stochastic projections using fitted distributions"""
    
    # Fit distributions to key variables
    claims_dist = fit_claims_distribution(df['Adjusted Claims'].values)
    
    # Calculate historical statistics
    inflation_mean = df['Inflation'].mean()
    inflation_std = df['Inflation'].std()
    
    # Fit age increment distribution
    age_increments = df['Average Age'].diff().dropna()
    age_increment_mean = age_increments.mean() if len(age_increments) > 0 else 0.5
    age_increment_std = age_increments.std() if len(age_increments) > 0 else 0.1
    
    # Run simulations
    simulation_results = []
    
    for sim in range(n_simulations):
        sim_data = []
        current_age = df.iloc[-1]['Average Age']
        current_members = df.iloc[-1]['Members']
        
        # Calculate base claims using historical average with some variability
        base_claims_mean = df['Adjusted Claims'].mean()
        base_claims_std = df['Adjusted Claims'].std()
        
        for year_idx, year in enumerate(projection_years):
            # Sample from distributions
            inflation = np.random.normal(inflation_mean, inflation_std)
            age_increment = max(0, np.random.normal(age_increment_mean, age_increment_std))
            
            # Calculate stochastic claims
            if year_idx == 0:
                # First year - use weighted average from historical data with some noise
                base_claims = base_claims_mean + np.random.normal(0, base_claims_std * 0.1)
            else:
                # Subsequent years - apply growth with stochastic component
                base_claims = sim_data[-1]['claims'] / sim_data[-1]['medical_trend'] / sim_data[-1]['aging_factor']
            
            # Apply trends
            current_age += age_increment
            aging_factor = calculate_member_risk_trend(df.iloc[-1]['Average Age'], current_age)
            medical_trend = (1 + inflation) ** (year_idx + 1)
            
            # Add random shock
            shock = np.random.normal(1.0, 0.05)  # 5% standard deviation
            
            projected_claims = base_claims * aging_factor * medical_trend * shock
            pmpm = projected_claims / (12 * current_members)
            
            sim_data.append({
                'year': year,
                'claims': projected_claims,
                'pmpm': pmpm,
                'pmpy': pmpm * 12,
                'age': current_age,
                'inflation': inflation,
                'aging_factor': aging_factor,
                'medical_trend': medical_trend
            })
        
        simulation_results.append(sim_data)
    
    return simulation_results

def calculate_risk_metrics(simulation_results, confidence_levels=[0.95, 0.99]):
    """Calculate VaR and CVaR for claims projections"""
    
    # Extract final year claims from all simulations
    final_pmpm = [sim[-1]['pmpm'] for sim in simulation_results]
    
    risk_metrics = {}
    
    for confidence in confidence_levels:
        # Value at Risk (upper tail for costs)
        var = np.percentile(final_pmpm, confidence * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        worst_cases = [c for c in final_pmpm if c >= var]
        cvar = np.mean(worst_cases) if worst_cases else var
        
        risk_metrics[f'VaR_{int(confidence*100)}'] = var
        risk_metrics[f'CVaR_{int(confidence*100)}'] = cvar
    
    return risk_metrics

def plot_stochastic_results(simulation_results, projection_years):
    """Create visualizations for stochastic results"""
    
    # Extract data for plotting
    n_years = len(projection_years)
    n_sims = len(simulation_results)
    
    pmpm_matrix = np.zeros((n_sims, n_years))
    for i, sim in enumerate(simulation_results):
        for j, year_data in enumerate(sim):
            pmpm_matrix[i, j] = year_data['pmpm']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Fan chart
    ax1 = plt.subplot(2, 2, 1)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(percentiles)))
    
    for i, p in enumerate(percentiles):
        values = np.percentile(pmpm_matrix, p, axis=0)
        ax1.plot(projection_years, values, color=colors[i], 
                label=f'{p}th percentile', linewidth=2 if p == 50 else 1)
    
    ax1.fill_between(projection_years, 
                    np.percentile(pmpm_matrix, 5, axis=0),
                    np.percentile(pmpm_matrix, 95, axis=0),
                    alpha=0.2, color='blue', label='90% Confidence Interval')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('PMPM ($)')
    ax1.set_title('Stochastic Projection Fan Chart')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Distribution of final year
    ax2 = plt.subplot(2, 2, 2)
    final_pmpm = pmpm_matrix[:, -1]
    ax2.hist(final_pmpm, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(final_pmpm), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: ${np.mean(final_pmpm):.2f}')
    ax2.axvline(np.percentile(final_pmpm, 95), color='orange', 
               linestyle='--', linewidth=2, label=f'95th %ile: ${np.percentile(final_pmpm, 95):.2f}')
    ax2.set_xlabel(f'PMPM in {projection_years[-1]} ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of {projection_years[-1]} PMPM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Year-over-year growth distribution
    ax3 = plt.subplot(2, 2, 3)
    growth_rates = []
    for sim in simulation_results:
        if len(sim) > 1:
            growth = (sim[-1]['pmpm'] / sim[0]['pmpm']) ** (1/(len(sim)-1)) - 1
            growth_rates.append(growth * 100)
    
    ax3.hist(growth_rates, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(growth_rates), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(growth_rates):.1f}%')
    ax3.set_xlabel('Annualized Growth Rate (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Annualized PMPM Growth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Correlation matrix
    ax4 = plt.subplot(2, 2, 4)
    # Extract final year metrics
    final_metrics = pd.DataFrame([{
        'PMPM': sim[-1]['pmpm'],
        'Age': sim[-1]['age'],
        'Inflation': sim[-1]['inflation']
    } for sim in simulation_results])
    
    corr_matrix = final_metrics.corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45)
    ax4.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center')
    
    ax4.set_title('Correlation Matrix')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    return fig

def show_calculation_details(scenario_name, scenario_data, original_df, scenario_params):
    """Show detailed calculation breakdown for a scenario"""
    df_summary, pmpm_first_year, pmpy_first_year, df_future = scenario_data
    infl_adj, age_strategy = scenario_params
    
    st.write(f"#### {scenario_name} - Step-by-Step Calculations")
    
    # Show scenario parameters
    st.write("**Scenario Parameters:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Inflation Adjustment", f"{infl_adj:+.1%}")
    with col2:
        age_desc = {
            'increment_0.5': '+0.5 years annually',
            'increment_1.0': '+1.0 year annually', 
            'constant': 'No change'
        }
        st.metric("Age Strategy", age_desc.get(age_strategy, age_strategy))
    
    # Step 1: Historical Data Adjustments
    st.write("**Step 1: Historical Data Adjustments**")
    adjustment_df = original_df[['Year', 'Total Claims', 'Months', 'IBNR_Months', 'High-Cost', 'Non-Recurring']].copy()
    adjustment_df['Annualized Claims'] = original_df['Annualized Claims']
    adjustment_df['IBNR Amount'] = original_df['IBNR Amount']
    adjustment_df['Adjusted Claims'] = (adjustment_df['Annualized Claims'] + 
                                      adjustment_df['IBNR Amount'] - 
                                      original_df['Annualized High-Cost'] - 
                                      original_df['Annualized Non-Recurring'])
    
    # Format for display
    display_adj = adjustment_df.copy()
    for col in ['Total Claims', 'Annualized Claims', 'IBNR Amount', 'Adjusted Claims']:
        display_adj[col] = display_adj[col].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(display_adj)
    
    # Step 2: Trend Calculations
    st.write("**Step 2: Trend Factor Calculations**")
    
    latest_age = original_df.iloc[-1]["Average Age"]
    projected_age = latest_age + (0.5 if age_strategy == 'increment_0.5' else 1.0 if age_strategy == 'increment_1.0' else 0)
    latest_inflation = original_df.iloc[-1]["Inflation"] + infl_adj
    
    st.write(f"- **Latest Historical Age:** {latest_age}")
    st.write(f"- **Projected Age:** {projected_age}")
    st.write(f"- **Adjusted Inflation Rate:** {latest_inflation:.1%}")
    
    # Calculate and show trends for each year
    trend_df = original_df[['Year', 'Average Age', 'Inflation']].copy()
    trend_df['Adjusted Inflation'] = trend_df['Inflation'] + infl_adj
    trend_df['Years to Project'] = max(original_df['Year']) + 1 - trend_df['Year']
    
    # Calculate medical trend
    medical_trends = []
    member_risk_trends = []
    
    for i, row in trend_df.iterrows():
        # Medical trend calculation
        trend = 1.0
        years_to_project = int(row['Years to Project'])
        for year_offset in range(years_to_project):
            if i + year_offset < len(trend_df):
                inflation_rate = trend_df.iloc[i + year_offset]['Adjusted Inflation']
            else:
                inflation_rate = latest_inflation
            trend *= (1 + inflation_rate)
        medical_trends.append(trend)
        
        # Member risk trend
        risk_trend = calculate_member_risk_trend(row['Average Age'], projected_age)
        member_risk_trends.append(risk_trend)
    
    trend_df['Medical Trend'] = medical_trends
    trend_df['Member Risk Trend'] = member_risk_trends
    trend_df['Combined Trend'] = trend_df['Medical Trend'] * trend_df['Member Risk Trend']
    
    # Format for display
    display_trend = trend_df.copy()
    display_trend['Adjusted Inflation'] = display_trend['Adjusted Inflation'].apply(lambda x: f"{x:.1%}")
    display_trend['Medical Trend'] = display_trend['Medical Trend'].apply(lambda x: f"{x:.4f}")
    display_trend['Member Risk Trend'] = display_trend['Member Risk Trend'].apply(lambda x: f"{x:.4f}")
    display_trend['Combined Trend'] = display_trend['Combined Trend'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_trend)
    
    # Step 3: Projected Claims Calculation
    st.write("**Step 3: Projected Claims & PMPM Calculation**")
    
    projection_df = original_df[['Year', 'Members', 'Weight']].copy()
    projection_df['Adjusted Claims'] = adjustment_df['Adjusted Claims']
    projection_df['Combined Trend'] = trend_df['Combined Trend']
    projection_df['Projected Claims'] = projection_df['Adjusted Claims'] * projection_df['Combined Trend']
    projection_df['PMPM'] = projection_df['Projected Claims'] / (12 * projection_df['Members'])
    projection_df['PMPY'] = projection_df['PMPM'] * 12
    
    # Format for display
    display_proj = projection_df.copy()
    for col in ['Adjusted Claims', 'Projected Claims']:
        display_proj[col] = display_proj[col].apply(lambda x: f"{x:,.0f}")
    display_proj['Combined Trend'] = display_proj['Combined Trend'].apply(lambda x: f"{x:.4f}")
    display_proj['PMPM'] = display_proj['PMPM'].apply(lambda x: f"{x:.2f}")
    display_proj['PMPY'] = display_proj['PMPY'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_proj)
    
    # Step 4: Weighted Average Calculation
    st.write("**Step 4: Weighted Average PMPM Calculation**")
    
    weighted_calculation = projection_df[['Year', 'PMPM', 'Weight']].copy()
    weighted_calculation['PMPM × Weight'] = weighted_calculation['PMPM'] * weighted_calculation['Weight']
    
    total_weighted_pmpm = weighted_calculation['PMPM × Weight'].sum()
    total_weight = weighted_calculation['Weight'].sum()
    weighted_avg_pmpm = total_weighted_pmpm / total_weight
    
    # Format for display
    display_weighted = weighted_calculation.copy()
    display_weighted['PMPM'] = display_weighted['PMPM'].apply(lambda x: f"{x:.2f}")
    display_weighted['PMPM × Weight'] = display_weighted['PMPM × Weight'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_weighted)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Weighted PMPM", f"{total_weighted_pmpm:.2f}")
    with col2:
        st.metric("Total Weight", f"{total_weight:.1f}")
    with col3:
        st.metric("Weighted Average PMPM", f"{weighted_avg_pmpm:.2f}")
    
    # Step 5: Future Year Projections
    st.write("**Step 5: Future Year Projections**")
    
    future_calc = df_future.copy()
    future_calc['Total Claims'] = future_calc['PMPY'] * original_df.iloc[-1]['Members']
    
    # Show year-over-year changes
    if len(future_calc) > 1:
        future_calc['PMPM Growth'] = future_calc['PMPM'].pct_change()
        future_calc['Claims Growth'] = future_calc['Total Claims'].pct_change()
    
    # Format for display
    display_future = future_calc.copy()
    display_future['PMPM'] = display_future['PMPM'].apply(lambda x: f"{x:.2f}")
    display_future['PMPY'] = display_future['PMPY'].apply(lambda x: f"{x:.2f}")
    display_future['Total Claims'] = display_future['Total Claims'].apply(lambda x: f"{x:,.0f}")
    
    if 'PMPM Growth' in display_future.columns:
        display_future['PMPM Growth'] = display_future['PMPM Growth'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        display_future['Claims Growth'] = display_future['Claims Growth'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    
    st.dataframe(display_future)
    
    # Show aging factor examples
    if age_strategy != 'constant':
        st.write("**Aging Factor Examples:**")
        age_examples = []
        current_age = latest_age
        for i in range(3):
            next_age = current_age + (0.5 if age_strategy == 'increment_0.5' else 1.0)
            current_factor = get_aging_factor(current_age)
            next_factor = get_aging_factor(next_age)
            risk_trend = next_factor / current_factor if current_factor > 0 else 1.0
            
            age_examples.append({
                'From Age': f"{current_age:.1f}",
                'To Age': f"{next_age:.1f}",
                'Current Factor': f"{current_factor:.4f}",
                'Next Factor': f"{next_factor:.4f}",
                'Risk Trend': f"{risk_trend:.4f}"
            })
            current_age = next_age
        
        st.dataframe(pd.DataFrame(age_examples))

st.title("📊 Enhanced Medical Claims Projection with Stochastic Analysis")

st.markdown("""
**Upload your Excel/CSV file with these columns:**

**Required:** `Year, Total Claims, Members, Average Age`

**Optional:** `High-Cost, Non-Recurring, IBNR_Months, Inflation, Weight, Months`

*Note: IBNR_Months = number of months of claims lag (e.g., 1.5 = 1.5 months of average claims)*

*Note: If optional columns are missing, default values will be used*

*Note: This tool dynamically handles any number of historical years (2, 3, 5, etc.)*
""")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("### 📋 Raw Input Data")
        st.dataframe(df)
        
        # Validate required columns
        required_cols = ['Year', 'Total Claims', 'Members', 'Average Age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Check number of years and display info
        num_years = len(df)
        st.info(f"📊 Processing {num_years} years of historical data ({df['Year'].min()} - {df['Year'].max()})")
        
        # Handle missing columns with defaults
        if 'Months' not in df.columns:
            df['Months'] = 12  # Default to 12 full months
            st.info("ℹ️ No 'Months' column found. Assuming all data is for 12 full months.")
        
        if 'IBNR_Months' not in df.columns:
            df['IBNR_Months'] = 0
            st.info("ℹ️ No 'IBNR_Months' column found. Using 0 months lag for all years.")
        
        if 'High-Cost' not in df.columns:
            df['High-Cost'] = 0
            st.info("ℹ️ No 'High-Cost' column found. Using 0 for all years.")
        
        if 'Non-Recurring' not in df.columns:
            df['Non-Recurring'] = 0
            st.info("ℹ️ No 'Non-Recurring' column found. Using 0 for all years.")
        
        if 'Inflation' not in df.columns:
            df['Inflation'] = 0.05  # Default 5%
            st.info("ℹ️ No 'Inflation' column found. Using 5% for all years.")
        
        if 'Weight' not in df.columns:
            df['Weight'] = 1.0  # Equal weighting
            st.info("ℹ️ No 'Weight' column found. Using equal weighting for all years.")
        
        # Annualize data if needed
        df = df.sort_values('Year')
        df['Annualized Claims'] = df['Total Claims'] * (12 / df['Months'])
        
        # Calculate IBNR amount based on months of average claims
        df['Monthly Average Claims'] = df['Annualized Claims'] / 12
        df['IBNR Amount'] = df['Monthly Average Claims'] * df['IBNR_Months']
        
        df['Annualized High-Cost'] = df['High-Cost'] * (12 / df['Months'])
        df['Annualized Non-Recurring'] = df['Non-Recurring'] * (12 / df['Months'])
        
        # Calculate adjusted claims for stochastic analysis
        df['Adjusted Claims'] = (df['Annualized Claims'] + 
                                df['IBNR Amount'] - 
                                df['Annualized High-Cost'] - 
                                df['Annualized Non-Recurring'])
        
        # Show IBNR calculation if there's any IBNR
        if (df['IBNR_Months'] > 0).any():
            st.write("### 🔄 IBNR Calculation")
            ibnr_info = df[['Year', 'IBNR_Months', 'Monthly Average Claims', 'IBNR Amount']].copy()
            ibnr_info['Monthly Average Claims'] = ibnr_info['Monthly Average Claims'].apply(lambda x: f"{x:,.0f}")
            ibnr_info['IBNR Amount'] = ibnr_info['IBNR Amount'].apply(lambda x: f"{x:,.0f}")
            ibnr_info['IBNR_Months'] = ibnr_info['IBNR_Months'].apply(lambda x: f"{x} months" if x != 0 else "No lag")
            st.dataframe(ibnr_info)

        # Show data period info if there's partial year data
        if (df['Months'] != 12).any():
            st.write("### 📊 Data Period Analysis")
            period_info = df[['Year', 'Months', 'Total Claims', 'Annualized Claims']].copy()
            period_info['Period'] = period_info['Months'].apply(lambda x: f"{x} months" if x != 12 else "Full year")
            period_info['Total Claims'] = period_info['Total Claims'].apply(lambda x: f"{x:,.0f}")
            period_info['Annualized Claims'] = period_info['Annualized Claims'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(period_info[['Year', 'Period', 'Total Claims', 'Annualized Claims']])

        # Dynamic projection years based on current year
        current_year = 2024  # You can make this dynamic with datetime.now().year
        max_historical_year = int(df['Year'].max())
        
        # Generate projection years starting from the year after the latest historical data
        projection_start_year = max_historical_year + 1
        projection_years = list(range(projection_start_year, projection_start_year + 3))  # Next 3 years
        
        st.info(f"🔮 Projecting for years: {', '.join(map(str, projection_years))}")

        def run_projection(df, inflation_adj=0.0, age_strategy='increment_0.5'):
            df = df.copy()
            latest_age = df.iloc[-1]["Average Age"]
            projected_age = latest_age + (0.5 if age_strategy == 'increment_0.5' else 1.0 if age_strategy == 'increment_1.0' else 0)
            latest_inflation = df.iloc[-1]["Inflation"] + inflation_adj
            members = df.iloc[-1]["Members"]

            # Calculate trends for each historical year
            for i in df.index:
                # Use annualized values and add IBNR amount, subtract high-cost and non-recurring
                df.at[i, "Adjusted Claims"] = (df.at[i, "Annualized Claims"] + 
                                             df.at[i, "IBNR Amount"] - 
                                             df.at[i, "Annualized High-Cost"] - 
                                             df.at[i, "Annualized Non-Recurring"])
                
                # Calculate medical trend from this year to projection year
                trend = 1.0
                years_to_project = projection_start_year - df.at[i, "Year"]
                for year_offset in range(years_to_project):
                    if year_offset < len(df) - i:
                        inflation_rate = df.iloc[i + year_offset]["Inflation"] + inflation_adj
                    else:
                        inflation_rate = latest_inflation
                    trend *= (1 + inflation_rate)
                
                # Calculate member risk trend
                risk_trend = calculate_member_risk_trend(df.at[i, "Average Age"], projected_age)
                
                df.at[i, "Medical Trend to PY"] = round(trend, 8)
                df.at[i, "Member Risk Trend to PY"] = risk_trend
                df.at[i, "Combined Trend Factor"] = round(trend * risk_trend, 8)
                df.at[i, "Projected Trended Claims"] = round(df.at[i, "Adjusted Claims"] * df.at[i, "Combined Trend Factor"], 2)
                df.at[i, "Projected PMPM"] = round(df.at[i, "Projected Trended Claims"] / (12 * df.at[i, "Members"]), 2)
                df.at[i, "Projected PMPY"] = round(df.at[i, "Projected PMPM"] * 12, 2)

            # Calculate weighted average PMPM for the first projection year
            weighted_pmpm = (df["Projected PMPM"] * df["Weight"]).sum() / df["Weight"].sum()

            # Generate future projections for all projection years
            results = []
            base_pmpm = weighted_pmpm
            age = projected_age
            
            for year in projection_years:
                if year == projection_years[0]:
                    # First projection year - use the weighted average
                    next_age = age
                    risk_trend = None
                    inflation_factor = None
                else:
                    # Subsequent years - apply trends
                    next_age = age + (0.5 if age_strategy == 'increment_0.5' else 1.0 if age_strategy == 'increment_1.0' else 0)
                    risk_trend = calculate_member_risk_trend(age, next_age)
                    inflation_factor = 1 + latest_inflation
                    base_pmpm = base_pmpm * inflation_factor * risk_trend
                
                pmpy = base_pmpm * 12
                results.append({
                    "Year": year,
                    "Projected Age": round(next_age, 1),
                    "Risk Trend": round(risk_trend, 6) if risk_trend is not None else None,
                    "Inflation Factor": round(inflation_factor, 6) if inflation_factor is not None else None,
                    "PMPM": round(base_pmpm, 2),
                    "PMPY": round(pmpy, 2)
                })
                age = next_age

            return df, weighted_pmpm, weighted_pmpm * 12, pd.DataFrame(results)

        # Define the 3 scenarios
        scenarios = {
            "Scenario 1 - Base Case": (0.0, 'increment_0.5'),
            "Scenario 2 - Lower Inflation, Static Age": (-0.01, 'constant'),
            "Scenario 3 - Higher Inflation, Fast Aging": (0.01, 'increment_1.0')
        }

        st.write("### 🎯 Scenario Analysis")
        st.markdown(f"""
        - **Scenario 1 (Base Case):** Current inflation rate, age increases by 0.5 years annually
        - **Scenario 2 (Conservative):** 1% lower inflation, age remains constant
        - **Scenario 3 (Aggressive):** 1% higher inflation, age increases by 1.0 year annually
        
        *Projecting from {num_years} years of historical data ({df['Year'].min()}-{df['Year'].max()}) to {projection_years[0]}-{projection_years[-1]}*
        """)

        scenario_results = {}
        for name, (infl_adj, age_mode) in scenarios.items():
            df_out, pmpm_first_year, pmpy_first_year, future_proj = run_projection(df, infl_adj, age_mode)
            scenario_results[name] = (df_out[["Year", "Months", "Adjusted Claims", "Projected PMPM", "Projected PMPY"]], 
                                    pmpm_first_year, pmpy_first_year, future_proj)

        # Display results for each scenario
        for name, (df_summary, pmpm_first_year, pmpy_first_year, df_future) in scenario_results.items():
            st.write(f"### ✅ {name}")
            
            # Show historical analysis
            st.write("**Historical Analysis:**")
            display_df = df_summary.copy()
            display_df['Adjusted Claims'] = display_df['Adjusted Claims'].apply(lambda x: f"{x:,.0f}")
            display_df['Period'] = display_df['Months'].apply(lambda x: f"{x}M" if x != 12 else "12M")
            st.dataframe(display_df[['Year', 'Period', 'Adjusted Claims', 'Projected PMPM', 'Projected PMPY']])
            
            st.write(f"📈 **{projection_years[0]} PMPM:** {round(pmpm_first_year, 2)} | **PMPY:** {round(pmpy_first_year, 2)}")
            
            # Show future projections
            st.write("**Future Projections:**")
            st.dataframe(df_future)

        # Comparison Chart - Dynamic based on projection years
        st.write(f"### 📊 Scenario Comparison: Total Claims Projection ({projection_years[0]}-{projection_years[-1]})")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (name, (_, _, _, df_future)) in enumerate(scenario_results.items()):
            total_claims = df_future["PMPY"] * df.iloc[-1]["Members"]
            ax.plot(df_future["Year"], total_claims, marker='o', linewidth=2, 
                   label=name, color=colors[i], markersize=8)
            
            # Add value annotations
            for _, row in df_future.iterrows():
                total = row['PMPY'] * df.iloc[-1]['Members']
                ax.annotate(f"{int(total):,}", 
                           (row["Year"], total),
                           textcoords="offset points", xytext=(0, 10), 
                           ha='center', fontsize=9, color=colors[i])
        
        ax.set_xticks(df_future["Year"])
        ax.set_xticklabels([int(year) for year in df_future["Year"]])
        ax.set_title(f"Total Projected Claims by Scenario ({projection_years[0]}-{projection_years[-1]})", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel("Total Claims (SGD)")
        ax.set_xlabel("Policy Year")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis to show values in millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        st.pyplot(fig)

        # Add Stochastic Analysis Section
        st.write("---")
        st.write("### 🎲 Stochastic Analysis")
        
        st.markdown("""
        Stochastic modeling incorporates uncertainty and randomness into projections, providing probability distributions 
        instead of single point estimates. This helps quantify risk and understand the range of possible outcomes.
        """)
        
        # Stochastic analysis parameters
        col1, col2 = st.columns(2)
        with col1:
            n_simulations = st.number_input("Number of Simulations", 
                                          min_value=1000, 
                                          max_value=50000, 
                                          value=10000, 
                                          step=1000,
                                          help="More simulations provide better accuracy but take longer")
        
        with col2:
            stochastic_scenario = st.selectbox("Select Base Scenario for Stochastic Analysis",
                                             list(scenarios.keys()),
                                             help="The deterministic scenario to use as the base for stochastic modeling")
        
        if st.button("🎯 Run Stochastic Analysis", type="primary"):
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                # Run stochastic projections
                simulation_results = stochastic_projection(df, projection_years, n_simulations)
                
                # Display results
                st.success(f"✅ Completed {n_simulations:,} simulations!")
                
                # Show risk metrics
                st.write("#### 📊 Risk Metrics")
                risk_metrics = calculate_risk_metrics(simulation_results)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("95% VaR (PMPM)", 
                             f"${risk_metrics['VaR_95']:.2f}",
                             help="95% of simulations resulted in PMPM below this value")
                with col2:
                    st.metric("95% CVaR (PMPM)", 
                             f"${risk_metrics['CVaR_95']:.2f}",
                             help="Average PMPM in the worst 5% of scenarios")
                with col3:
                    st.metric("99% VaR (PMPM)", 
                             f"${risk_metrics['VaR_99']:.2f}",
                             help="99% of simulations resulted in PMPM below this value")
                with col4:
                    st.metric("99% CVaR (PMPM)", 
                             f"${risk_metrics['CVaR_99']:.2f}",
                             help="Average PMPM in the worst 1% of scenarios")
                
                # Plot stochastic results
                st.write("#### 📈 Stochastic Projection Visualizations")
                fig = plot_stochastic_results(simulation_results, projection_years)
                st.pyplot(fig)
                
                # Summary statistics table
                st.write("#### 📊 Summary Statistics by Year")
                
                summary_stats = []
                for year_idx, year in enumerate(projection_years):
                    year_pmpm = [sim[year_idx]['pmpm'] for sim in simulation_results]
                    
                    summary_stats.append({
                        'Year': year,
                        'Mean PMPM': f"${np.mean(year_pmpm):.2f}",
                        'Median PMPM': f"${np.median(year_pmpm):.2f}",
                        'Std Dev': f"${np.std(year_pmpm):.2f}",
                        '5th Percentile': f"${np.percentile(year_pmpm, 5):.2f}",
                        '95th Percentile': f"${np.percentile(year_pmpm, 95):.2f}",
                        'CV': f"{(np.std(year_pmpm) / np.mean(year_pmpm)):.1%}"
                    })
                
                st.dataframe(pd.DataFrame(summary_stats))
                
                # Probability analysis
                st.write("#### 🎯 Probability Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    threshold_type = st.selectbox("Threshold Type", ["PMPM", "Total Claims"])
                
                with col2:
                    if threshold_type == "PMPM":
                        default_threshold = int(np.mean([sim[-1]['pmpm'] for sim in simulation_results]) * 1.2)
                        threshold = st.number_input("PMPM Threshold ($)", 
                                                  value=default_threshold,
                                                  help="Calculate probability of exceeding this PMPM")
                    else:
                        default_threshold = int(np.mean([sim[-1]['pmpm'] for sim in simulation_results]) * 12 * df.iloc[-1]['Members'] * 1.2)
                        threshold = st.number_input("Total Claims Threshold ($)", 
                                                  value=default_threshold,
                                                  help="Calculate probability of exceeding this total claims amount")
                
                # Calculate probabilities
                if threshold_type == "PMPM":
                    final_values = [sim[-1]['pmpm'] for sim in simulation_results]
                else:
                    final_values = [sim[-1]['pmpm'] * 12 * df.iloc[-1]['Members'] for sim in simulation_results]
                
                prob_exceed = sum(1 for v in final_values if v > threshold) / len(final_values)
                
                st.metric(f"Probability of exceeding ${threshold:,.0f} {threshold_type} in {projection_years[-1]}", 
                         f"{prob_exceed:.1%}")
                
                # Scenario comparison with stochastic
                st.write("#### 🔄 Stochastic vs Deterministic Comparison")
                
                # Get deterministic results
                base_scenario_params = scenarios[stochastic_scenario]
                _, det_pmpm, _, det_future = run_projection(df, base_scenario_params[0], base_scenario_params[1])
                
                comparison_data = []
                for year_idx, year in enumerate(projection_years):
                    stoch_pmpm = [sim[year_idx]['pmpm'] for sim in simulation_results]
                    det_pmpm_value = det_future.iloc[year_idx]['PMPM']
                    
                    comparison_data.append({
                        'Year': year,
                        'Deterministic PMPM': f"${det_pmpm_value:.2f}",
                        'Stochastic Mean': f"${np.mean(stoch_pmpm):.2f}",
                        'Difference': f"{((np.mean(stoch_pmpm) - det_pmpm_value) / det_pmpm_value):.1%}",
                        '90% CI': f"${np.percentile(stoch_pmpm, 5):.2f} - ${np.percentile(stoch_pmpm, 95):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data))
                
                # Download stochastic results
                st.write("#### 💾 Export Stochastic Results")
                
                # Prepare data for export
                export_data = []
                for sim_idx, sim in enumerate(simulation_results[:1000]):  # Limit to 1000 for file size
                    for year_idx, year_data in enumerate(sim):
                        export_data.append({
                            'Simulation': sim_idx + 1,
                            'Year': year_data['year'],
                            'PMPM': year_data['pmpm'],
                            'PMPY': year_data['pmpy'],
                            'Total Claims': year_data['pmpy'] * df.iloc[-1]['Members'],
                            'Age': year_data['age'],
                            'Inflation': year_data['inflation']
                        })
                
                export_df = pd.DataFrame(export_data)
                csv_stochastic = export_df.to_csv(index=False)
                
                st.download_button(
                    "📥 Download Stochastic Results (First 1,000 simulations)",
                    data=csv_stochastic,
                    file_name=f"stochastic_results_{n_simulations}_simulations.csv",
                    mime="text/csv"
                )

        # Show detailed calculations
        st.write("### 🧮 Detailed Calculation Breakdown")
        
        # Create tabs for each scenario
        tab1, tab2, tab3 = st.tabs(["Scenario 1 - Base Case", "Scenario 2 - Conservative", "Scenario 3 - Aggressive"])
        
        with tab1:
            show_calculation_details("Scenario 1 - Base Case", scenario_results["Scenario 1 - Base Case"], df, scenarios["Scenario 1 - Base Case"])
        
        with tab2:
            show_calculation_details("Scenario 2 - Lower Inflation, Static Age", scenario_results["Scenario 2 - Lower Inflation, Static Age"], df, scenarios["Scenario 2 - Lower Inflation, Static Age"])
        
        with tab3:
            show_calculation_details("Scenario 3 - Higher Inflation, Fast Aging", scenario_results["Scenario 3 - Higher Inflation, Fast Aging"], df, scenarios["Scenario 3 - Higher Inflation, Fast Aging"])

        # Summary comparison - Dynamic
        st.write("### 📋 Scenario Summary")
        summary_data = []
        for name, (_, pmpm_first_year, pmpy_first_year, df_future) in scenario_results.items():
            final_year = df_future.iloc[-1]
            first_year = df_future.iloc[0]
            total_first_year = first_year['PMPY'] * df.iloc[-1]["Members"]
            total_final_year = final_year['PMPY'] * df.iloc[-1]["Members"]
            
            years_span = len(projection_years)
            growth = (total_final_year - total_first_year) / total_first_year if total_first_year > 0 else 0
            
            summary_data.append({
                'Scenario': name,
                f'{projection_years[0]} Total Claims': f"{total_first_year:,.0f}",
                f'{projection_years[-1]} Total Claims': f"{total_final_year:,.0f}",
                f'{years_span-1}-Year Growth': f"{growth:.1%}",
                f'{projection_years[-1]} PMPY': f"{final_year['PMPY']:,.0f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)

        # Download option
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Enhanced Data", 
                          data=csv, 
                          file_name=f"enhanced_claims_projection_{num_years}_years.csv",
                          mime="text/csv")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Make sure your file has the required columns: Year, Total Claims, Members, Average Age")
        st.write("Debug info:")
        st.write(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.write("### 📝 Sample Data Formats")
    
    st.write("**Example 1: 2 Years of Data**")
    sample_data_2y = pd.DataFrame({
        'Year': [2022, 2023],
        'Total Claims': [4600000, 2650000],
        'IBNR_Months': [1.0, 2.0],
        'High-Cost': [250000, 150000],
        'Non-Recurring': [120000, 80000],
        'Members': [1050, 1100],
        'Average Age': [43.0, 43.5],
        'Inflation': [0.05, 0.06],
        'Weight': [1.0, 1.0],
        'Months': [12, 6]
    })
    st.dataframe(sample_data_2y)
    
    st.write("**Example 2: 5 Years of Data**")
    sample_data_5y = pd.DataFrame({
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Total Claims': [3800000, 4200000, 4600000, 5000000, 2650000],
        'IBNR_Months': [1.0, 1.5, 1.5, 1.0, 2.0],
        'High-Cost': [180000, 200000, 250000, 280000, 150000],
        'Non-Recurring': [90000, 100000, 120000, 140000, 80000],
        'Members': [950, 1000, 1050, 1100, 1150],
        'Average Age': [41.5, 42.0, 42.5, 43.0, 43.5],
        'Inflation': [0.03, 0.04, 0.04, 0.05, 0.06],
        'Weight': [0.8, 0.9, 1.0, 1.0, 1.0],
        'Months': [12, 12, 12, 12, 6]
    })
    st.dataframe(sample_data_5y)
    
    st.caption("💡 Required: Year, Total Claims, Members, Average Age. The tool automatically adapts to any number of years (2, 3, 5, etc.)")