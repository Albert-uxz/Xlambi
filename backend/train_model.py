import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


# =====================================================
# XLAMBI AI RISK MODEL — TRAINING DATA
# =====================================================

data = {

    "speed": [
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,

        10, 30, 50, 70, 90,
        20, 40, 60, 80, 100,

        15, 35, 55, 75, 95,
        25, 45, 65, 85, 105
    ],

    "accuracy": [
        5, 5, 10, 10, 15,
        15, 20, 20, 25, 30,

        5, 10, 10, 15, 20,
        5, 10, 15, 20, 25,

        10, 10, 15, 15, 20,
        10, 15, 15, 20, 25
    ],

    # 0 = outside risk zone
    # 1 = inside risk zone
    "risk_zone": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,

        0, 0, 1, 1, 1,
        0, 1, 1, 1, 1
    ],

    # 0 = normal weather
    # 1 = risky weather
    "weather_risk": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,

        1, 1, 1, 1, 1,
        0, 0, 1, 1, 1
    ],

    # 0 = low traffic
    # 1 = medium traffic
    # 2 = high traffic
    "traffic_level": [
        0, 0, 0, 1, 1,
        1, 2, 2, 2, 2,

        0, 1, 1, 2, 2,
        1, 1, 2, 2, 2,

        0, 0, 1, 1, 2,
        1, 1, 2, 2, 2
    ],

    # Target risk score
    "risk_score": [
        10, 15, 20, 30, 40,
        50, 65, 75, 85, 95,

        25, 40, 50, 70, 85,
        35, 50, 65, 80, 95,

        30, 45, 60, 75, 90,
        25, 45, 65, 85, 100
    ]
}


# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(data)


# =====================================================
# INPUT FEATURES
# =====================================================

X = df[
    [
        "speed",
        "accuracy",
        "risk_zone",
        "weather_risk",
        "traffic_level"
    ]
]


# =====================================================
# TARGET
# =====================================================

y = df["risk_score"]


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train
)


# =====================================================
# TEST MODEL
# =====================================================

predictions = model.predict(
    X_test
)


error = mean_absolute_error(
    y_test,
    predictions
)


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "risk_model.pkl"
)


# =====================================================
# OUTPUT
# =====================================================

print()
print("======================================")
print("XLAMBI AI RISK MODEL")
print("======================================")

print(
    "Training samples:",
    len(df)
)

print(
    "Features:",
    list(X.columns)
)

print(
    "Mean Absolute Error:",
    round(error, 2)
)

print()
print("Random Forest trained successfully!")
print("Model saved as: risk_model.pkl")

print("======================================")