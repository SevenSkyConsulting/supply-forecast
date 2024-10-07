import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
start_date = datetime(2021, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create DataFrame
df = pd.DataFrame(index=date_range, columns=['Product1', 'Product2', 'Product3'])

# Generate sample data for each product
for product in df.columns:
    # Base demand
    base_demand = np.random.randint(50, 200)

    # Trend
    trend = np.linspace(0, 20, len(df))

    # Seasonality (yearly pattern)
    seasonality = 15 * np.sin(np.arange(len(df)) * (2 * np.pi / 365))

    # Weekly pattern
    weekly_pattern = 10 * (df.index.dayofweek < 5)  # Higher sales on weekdays

    # Combine components
    demand = base_demand + trend + seasonality + weekly_pattern

    # Add noise
    noise = np.random.normal(0, 5, len(df))

    # Ensure all values are positive and round to integers
    df[product] = np.round(np.maximum(demand + noise, 0)).astype(int)

# Reset index to make Date a column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("Sample sales_data.csv has been generated.")