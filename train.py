"""Train wait-time regressor from synthetic data. Run once: python train.py"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

rng = np.random.default_rng(42)
N = 10_000

tokens_ahead = rng.integers(0, 40, N)
active_counters = rng.integers(1, 6, N)
hour = rng.integers(8, 20, N)
dow = rng.integers(0, 7, N)
avg_service = rng.uniform(3, 12, N)
recent_throughput = rng.integers(0, 60, N)

def peak(h, d):
    m = 1.0
    if 11 <= h <= 14: m += 0.35
    if 17 <= h <= 19: m += 0.45
    if d >= 5: m += 0.20
    return m

mult = np.vectorize(peak)(hour, dow)
base = (tokens_ahead * avg_service) / np.maximum(active_counters, 1)
wait = np.maximum(0, base * mult + rng.normal(0, 3, N))

df = pd.DataFrame({
    "tokens_ahead": tokens_ahead, "active_counters": active_counters,
    "hour_of_day": hour, "day_of_week": dow,
    "avg_service_time": avg_service, "recent_throughput": recent_throughput,
    "wait_minutes": wait,
})
df.to_csv("synthetic_queue_data.csv", index=False)

X, y = df.drop(columns=["wait_minutes"]), df["wait_minutes"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

m = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
m.fit(Xtr, ytr)
pred = m.predict(Xte)
print(f"MAE:  {mean_absolute_error(yte, pred):.2f} min")
print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.2f} min")
joblib.dump(m, "wait_time_model.pkl")
print("Saved wait_time_model.pkl")