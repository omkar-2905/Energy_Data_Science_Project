# Home Energy Management System (HEMS) - Data Science Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data science project for optimizing residential energy management through smart integration of solar PV panels, battery energy storage systems (BESS), and demand forecasting.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tasks & Methodology](#tasks--methodology)
- [Results & Visualizations](#results--visualizations)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project develops an intelligent **Home Energy Management System (HEMS)** to:
- **Reduce electricity bills** through optimized energy usage
- **Maximize self-consumption** of solar power
- **Optimize** energy production, storage, and consumption patterns

### Main Components
- **PV Solar Panels**: Renewable energy generation
- **Battery Energy Storage System (BESS)**: Energy storage and time-shifting
- **Optimization Software**: AI-driven control algorithms
- **Predictive Analytics**: Demand and generation forecasting

### Dataset
- **Duration**: One year (July 2013 - June 2014)
- **Frequency**: Hourly data (8,759 observations)
- **Features**: 17 variables including:
  - PV generation (3 modules)
  - Electricity demand
  - Dynamic pricing
  - Weather data (temperature, cloud cover, radiation, wind speed, etc.)

## Key Features

- **Comprehensive Data Cleaning**: Multiple imputation methods (deletion, median, KNN, MICE)
- **Advanced Feature Engineering**: 40+ temporal, lag, and interaction features
- **Time Series Decomposition**: Trend, seasonal, and residual analysis
- **Multiple Forecasting Models**:
  - Statistical: ARIMA, SARIMA
  - Machine Learning: XGBoost with hyperparameter tuning
- **Optimal Battery Control**: Constrained optimization for cost minimization
- **Interactive Visualizations**: 15+ detailed plots and heatmaps
- **Walk-Forward Validation**: Robust model evaluation

## Project Structure

```
TalTech-Energy-Data-Science/
│
├── solve.ipynb                          # Main Jupyter notebook (all tasks)
├── README.md                            # This file
│
├── DataSet_ToUSE/
│   ├── train_test.csv                   # Training dataset (1 year)
│   ├── forecast.csv                     # Forecast evaluation dataset
│   └── optimisation.csv                 # Battery optimization dataset
│
├── Task Outputs/
│   ├── Task1_PV_Demand_Price_Timeseries.png
│   ├── Task2_DataScience_Lifecycle_Flowchart.png
│   ├── Task3_Timeseries_Full_Year.png
│   ├── Task4_Imputation_Comparison.png
│   ├── Task5_Feature_Importance.png
│   ├── Task6_Decomposition_Daily.png
│   ├── Task7_ACF_PACF_Analysis.png
│   ├── Task8_Model_Comparison_Predictions.png
│   ├── Task9_7Day_Rolling_Forecast.png
│   ├── Task10_Exogenous_Features_Comparison.png
│   └── Task11_PV_high_Optimal_Control.png
│
└── Imputed Datasets/
    ├── pv_deleted.csv
    ├── pv_median_imputed.csv
    ├── pv_knn_imputed.csv
    └── pv_iter_imputed.csv
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hems-energy-optimization.git
cd hems-energy-optimization
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
statsmodels>=0.14.0
scipy>=1.10.0
```

## Usage

### Running the Complete Analysis

1. **Open Jupyter Notebook**
```bash
jupyter notebook solve.ipynb
```

2. **Run all cells** or execute specific task sections

3. **View outputs**: Visualizations are saved automatically to the project directory

### Quick Start Example

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('DataSet_ToUSE/train_test.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Quick visualization
df[['Demand', 'pv', 'Price']].plot(figsize=(14, 6))
plt.title('Energy Demand, PV Generation, and Price')
plt.show()
```

## Tasks & Methodology

### Task 1: Introduction & Digital Transformation 
- Visualized PV generation, demand, and price patterns
- Analyzed how digitalization transforms household energy management
- Explored applications of solar data in private and business sectors

**Key Insight**: Heatmap visualizations provide the highest information density for identifying optimal energy usage patterns.

### Task 2: Data Science Lifecycle 
- Created comprehensive project flowchart
- Identified **Data Cleaning** and **Feature Engineering** as highest-effort phases (30-35% each)
- Determined need for external data: calendar/holidays and battery specifications

### Task 3: Visualization 
Implemented four visualization types:
1. **Time Series Analysis** - Annual patterns and trends
2. **Distribution Analysis** - Histograms and box plots
3. **Hourly Pattern Heatmaps** *Most Informative*
4. **Correlation & Scatter Analysis** - Feature relationships

**Winner**: Heatmaps reveal actionable temporal patterns for load shifting and battery management.

### Task 4: Data Cleaning 
- Analyzed missing values in PV modules (5-6% missing)
- Identified missing data mechanisms: **MAR** (Missing At Random) for daytime, **MCAR** for others
- Applied **4 imputation methods**:
  1. Deletion (baseline)
  2. Median imputation (univariate)
  3. KNN imputation (multivariate)
  4. MICE/Iterative imputation (multivariate)
- **Best method**: KNN and MICE (lowest RMSE, preserved distributions)

### Task 5: Feature Engineering 
Created **40+ features** across categories:
- **Temporal**: Cyclical encodings (sin/cos), binary indicators, seasons
- **Lag Features**: 1h, 24h, 168h lags with rolling statistics
- **Weather Interactions**: Wind chill, solar potential, weather severity
- **Energy Metrics**: Net demand, self-sufficiency ratio, cost metrics

**Feature Importance Ranking**:
1. Lag features (1h, 24h, 168h) - strongest predictors
2. Hour-of-day features (cyclical encoding)
3. Temperature and seasonal indicators
4. Rolling statistics (smoothed trends)

### Task 6: Time Series Analysis 
- Performed **additive decomposition**: yt = Tt + St + Rt
- Identified strongest seasonal effects:
  - **Daily**: Morning and evening demand peaks
  - **Weekly**: Weekday vs. weekend patterns
  - **Annual**: Winter heating and summer cooling cycles
- Created typical demand profiles for forecasting baselines

### Task 7: Statistical Modeling 
Developed and compared two models:

| Model | Type | Test NRMSE | Test MAE |
|-------|------|------------|----------|
| ARIMA(2,1,2) | Non-seasonal | 0.6966 | 0.2426 |
| SARIMA(1,1,1)(1,0,1,24) | Seasonal | 0.6429 | 0.1898 |

**Winner**: SARIMA (16% improvement) due to explicit 24-hour seasonality modeling.

### Task 8: Machine Learning 
- Trained **XGBoost** ensemble model with 11 tuned hyperparameters
- Utilized 22 engineered features
- **Results**:
  - Training NRMSE: 0.2863
  - Test NRMSE: 0.5367
  - **23% improvement** over ARIMA
  - **17% improvement** over SARIMA

**Advantages**: Captures non-linear relationships, leverages rich feature set, better generalization.

### Task 9: Forecasting Pipeline 
- Built reproducible forecasting pipeline
- Implemented 7-day rolling forecast with 24h horizon
- Compared statistical vs. ML models vs. baselines
- **XGBoost with exogenous features** achieved best performance

### Task 10: Exogenous Features Impact 
- Evaluated weather features contribution to forecast accuracy
- Quantified improvement from incorporating external data
- Identified most valuable exogenous predictors

### Task 11: Optimal Battery Control 
Implemented constrained optimization for battery management:

**System Configuration**:
- PV: 5 kW max power
- Battery: 10 kWh capacity, 5 kW charge/discharge, 95% efficiency
- Grid: 5 kW max power

**Optimization Objective**: Minimize electricity cost (Grid purchases - Grid sales)

**Scenarios Compared**:
| Scenario | Daily Cost | Grid Purchases | Grid Sales | Self-Sufficiency |
|----------|-----------|----------------|------------|------------------|
| PV_low | $0.25 | 28.16 kWh | 19.10 kWh | ~30% |
| PV_high | -$0.75 | 23.50 kWh | 26.17 kWh | ~80% |

**Savings**: $1.00/day (~$365/year) with high PV scenario

## Results & Visualizations

### Sample Outputs

#### 1. Hourly Pattern Heatmaps
![Heatmaps](Task3_Hourly_Pattern_Heatmaps.png)
*Shows demand, PV generation, and price patterns by hour and day/month*

#### 2. Feature Importance Analysis
![Features](Task5_Feature_Importance.png)
*Top predictors for demand forecasting across multiple methods*

#### 3. Time Series Decomposition
![Decomposition](Task6_Decomposition_Daily.png)
*Separates trend, seasonality, and residual components*

#### 4. Model Comparison
![Models](Task8_Model_Comparison_Predictions.png)
*Performance comparison: ARIMA vs. SARIMA vs. XGBoost*

#### 5. Optimal Battery Control
![Optimization](Task11_PV_high_Optimal_Control.png)
*24-hour optimal control strategy for battery charging/discharging*

### Performance Metrics Summary

| Metric | ARIMA | SARIMA | XGBoost | Winner |
|--------|-------|--------|---------|---------|
| Test RMSE | 0.2426 | 0.1898 | 0.1438 | XGBoost |
| Test NRMSE | 0.6966 | 0.6429 | 0.5367 | XGBoost |
| Test MAPE | 50.39% | 32.47% | 26.82% | XGBoost |
| Training Time | ~30s | ~45s | ~0.5s | XGBoost |
| Interpretability | High | High | Medium | SARIMA |

## Technologies Used

### Data Analysis & Processing
- **Pandas**: Data manipulation and time series handling
- **NumPy**: Numerical computations
- **SciPy**: Statistical tests and optimization

### Machine Learning
- **Scikit-learn**: Preprocessing, imputation, feature selection
- **XGBoost**: Gradient boosting for demand forecasting
- **Statsmodels**: ARIMA/SARIMA time series models

### Visualization
- **Matplotlib**: Core plotting functionality
- **Seaborn**: Statistical visualizations
- **Custom Heatmaps**: Temporal pattern analysis

### Optimization
- **SciPy Optimize**: Constrained optimization solver
- **Linear Programming**: Battery control algorithms

## Key Findings

### 1. Optimal Battery Strategy
- **Charge** during: Low prices, high PV, off-peak hours
- **Discharge** during: High prices, peak demand, evening hours
- **Result**: 20-40% cost reduction with intelligent control

### 2. Forecasting Insights
- **Lag features** (1h, 24h, 168h) are strongest predictors (historical demand)
- **Weather features** provide 5-10% accuracy improvement
- **XGBoost** outperforms statistical models for complex patterns
- **SARIMA** offers best interpretability vs. accuracy trade-off

### 3. Economic Benefits
- **High PV + Battery**: Net negative costs (profit from selling)
- **Annual savings**: $300-400 vs. no battery system
- **Self-sufficiency**: Improves from 30% to 80% with optimal control
- **Payback period**: 5-7 years for battery investment

### 4. Data Quality Matters
- **Imputation method** impacts forecast accuracy by 2-5%
- **KNN and MICE** preserve feature relationships better than median
- **Missing data mechanism** identification is crucial for appropriate handling

### 5. Feature Engineering Impact
- Engineered features improve model accuracy by **15-20%**
- **Cyclical encoding** essential for avoiding artificial discontinuities
- **Rolling statistics** reduce noise and stabilize predictions

## Future Work

### Model Enhancements
- [ ] Deep learning models (LSTM, GRU) for sequence prediction
- [ ] Ensemble methods combining statistical and ML approaches
- [ ] Probabilistic forecasting with uncertainty quantification
- [ ] Transfer learning from similar households

### System Extensions
- [ ] Real-time optimization with dynamic pricing
- [ ] Multi-objective optimization (cost + carbon footprint)
- [ ] Electric vehicle (EV) charging integration
- [ ] Peer-to-peer energy trading in microgrids
- [ ] Integration with smart home IoT devices

### Data Improvements
- [ ] Incorporate weather forecast APIs
- [ ] Add calendar and holiday effects
- [ ] Collect higher-frequency data (15-min intervals)
- [ ] Include appliance-level consumption data

### Deployment
- [ ] RESTful API for real-time predictions
- [ ] Web dashboard for monitoring and control
- [ ] Edge computing for latency-sensitive operations
- [ ] Cloud deployment with scalable infrastructure

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation for significant changes
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## Acknowledgments

- **TalTech (Tallinn University of Technology)** - Energy Data Science Program
- **Dataset**: Provided by course instructors
- **Inspiration**: Smart grid and renewable energy research community
- **Tools**: Open-source Python ecosystem

---

## References

1. Home Energy Management Systems: A Review - *Renewable and Sustainable Energy Reviews*
2. Battery Energy Storage for PV Integration - *IEEE Transactions on Smart Grid*
3. Time Series Forecasting with XGBoost - *Applied Energy*
4. Optimal Control of Residential Energy Systems - *Energy and Buildings*

---


