import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_excel("Performance_DataSet_Analyse.xlsx")

df = df[
    [
        "Framework",
        "Items",
        "Scenario",
        "LOC",
        "Component_Count",
        "Component_Depth",
        "CPU Time (ms)",
        "JS Execution Time (ms)",
        "Memory (MB)"
    ]
]

# ==========================================
# 2. PREPARE FEATURES
# ==========================================

df["Framework"] = df["Framework"].map({
    "Angular": 0,
    "React": 1
})

df = pd.get_dummies(df, columns=["Scenario"])
df = df.fillna(0)

X = df.drop(columns=[
    "CPU Time (ms)",
    "JS Execution Time (ms)",
    "Memory (MB)"
])

y_cpu = df["CPU Time (ms)"]
y_js = df["JS Execution Time (ms)"]
y_mem = df["Memory (MB)"]

# ==========================================
# 3. TRAIN / TEST SPLIT (80 / 20)
# ==========================================

X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(
    X, y_cpu, test_size=0.2, random_state=42
)

_, _, y_js_train, y_js_test = train_test_split(
    X, y_js, test_size=0.2, random_state=42
)

_, _, y_mem_train, y_mem_test = train_test_split(
    X, y_mem, test_size=0.2, random_state=42
)

# ==========================================
# 4. MODELS
# ==========================================

lin_cpu = LinearRegression()
rf_cpu = RandomForestRegressor(n_estimators=300, random_state=42)

lin_js = LinearRegression()
rf_js = RandomForestRegressor(n_estimators=300, random_state=42)

lin_mem = LinearRegression()
rf_mem = RandomForestRegressor(n_estimators=300, random_state=42)

lin_cpu.fit(X_train, y_cpu_train)
rf_cpu.fit(X_train, y_cpu_train)

lin_js.fit(X_train, y_js_train)
rf_js.fit(X_train, y_js_train)

lin_mem.fit(X_train, y_mem_train)
rf_mem.fit(X_train, y_mem_train)

# ==========================================
# 5. EVALUATION
# ==========================================

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: R2 = {r2:.3f}, RMSE = {rmse:.1f}")

print("\nCPU PERFORMANCE\n")
evaluate("Linear Regression", lin_cpu, X_test, y_cpu_test)
evaluate("Random Forest", rf_cpu, X_test, y_cpu_test)

print("\nJS PERFORMANCE\n")
evaluate("Linear Regression", lin_js, X_test, y_js_test)
evaluate("Random Forest", rf_js, X_test, y_js_test)

print("\nMEMORY PERFORMANCE\n")
evaluate("Linear Regression", lin_mem, X_test, y_mem_test)
evaluate("Random Forest", rf_mem, X_test, y_mem_test)

# ==========================================
# 6. PREDICTION EXAMPLE
# ==========================================

sample = X.iloc[0:1].copy()

for col in sample.columns:
    if sample[col].dtype == "bool":
        sample[col] = False
    else:
        sample[col] = 0

# React
sample["Framework"] = 1          
sample["Items"] = 10000
sample["LOC"] = 247
sample["Component_Count"] = 5
sample["Component_Depth"] = 2

for col in sample.columns:
    if "S5" in col:
        sample[col] = True

pred_cpu = rf_cpu.predict(sample)[0]
pred_js = rf_js.predict(sample)[0]
pred_mem = rf_mem.predict(sample)[0]

print("\nPREDICTION EXAMPLE")
print(f"Predicted CPU Time: {pred_cpu:.1f} ms")
print(f"Predicted JS Time: {pred_js:.1f} ms")
print(f"Predicted Memory: {pred_mem:.1f} MB")
