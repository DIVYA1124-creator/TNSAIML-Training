import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = Path(__file__).resolve().parent
data = base_dir / "house_price_prediction_data.csv"
df = pd.read_csv(data)

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

df = df[df["price"] > 0]
df = df[df["price"] <= df["price"].quantile(0.99)]
df = df[(df["bedrooms"] > 0) & (df["bathrooms"] > 0)]
print(df.shape)

df = df.drop(["date", "street", "statezip", "price_per_sqft", "sqft_above"], axis=1)

x = df.drop("price", axis=1)
y = df["price"]

categorical_cols = ["city"]
numerical_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_basement",
    "yr_built", "yr_renovated"
]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Lasso Regression": Lasso(alpha=100.0, max_iter=20000),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
}

results = []
trained = {}

for name, algo in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", algo)])
    pipe.fit(x_train, y_train)
    train_pred = pipe.predict(x_train)
    test_pred = pipe.predict(x_test)
    cv_r2 = cross_val_score(pipe, x_train, y_train, cv=5, scoring="r2").mean()
    results.append({
        "Model": name,
        "Train R2": round(r2_score(y_train, train_pred), 3),
        "Test R2": round(r2_score(y_test, test_pred), 3),
        "CV R2": round(cv_r2, 3),
        "MAE": round(mean_absolute_error(y_test, test_pred)),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, test_pred)))
    })
    trained[name] = pipe

results_df = pd.DataFrame(results).sort_values("CV R2", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

best_name = results_df.loc[0, "Model"]
best_model = trained[best_name]
print("best model:", best_name)

feature_names = list(x.columns)

joblib.dump(best_model, base_dir.parent / "model.joblib")
joblib.dump(feature_names, base_dir.parent / "feature_names.joblib")

sample = pd.DataFrame([{
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 2000,
    "sqft_lot": 7500,
    "floors": 2.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_basement": 0,
    "yr_built": 1990,
    "yr_renovated": 0,
    "city": "Seattle"
}])

prediction = best_model.predict(sample[feature_names])[0]
print("predicted price:", round(prediction, 2))
