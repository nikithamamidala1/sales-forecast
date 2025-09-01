# sales_forecast_lr.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset (Month vs Sales)
df = pd.read_csv("sales.csv")

# Example CSV structure:
# Month,Sales
# 1,200
# 2,220
# 3,250
# ...

X = df[["Month"]]   # Feature
y = df["Sales"]     # Target

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future months (next 6 months)
future_months = pd.DataFrame({"Month": range(len(df)+1, len(df)+7)})
forecast = model.predict(future_months)

# Show results
print("Forecasted Sales for next 6 months:")
print(pd.DataFrame({"Month": future_months["Month"], "Forecast": forecast}))

# Plot
plt.plot(df["Month"], y, label="Actual Sales")
plt.plot(future_months["Month"], forecast, label="Forecast", linestyle="--")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()
