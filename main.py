import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datenimport
df = pd.read_excel("main.xlsx", sheet_name="YK Basistabelle")

# Spaltenauswahl und Datentypumwandlung
df = df[["Material", "Datum", "Menge"]]
df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
df['Material'] = df['Material'].astype(str)
df = df[df['Material'].str.isnumeric()]
df['Material'] = df['Material'].astype(int)

# Datenbereinigung
df = df.dropna(subset=["Datum", "Menge"])

# Umwandlung von 'Datum' in Unix-Zeitstempel
df['Datum'] = df['Datum'].astype('int64') // 10**9

# Datenaufteilung
X = df[["Material", "Datum"]]
y = df["Menge"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modelltraining
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen und Leistungsbewertung
y_pred = model.predict(X_test)
print("MSE: ", mean_squared_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

# Umwandlung von 'Datum' in Unix-Zeitstempel

# Features und Zielvariable
X = df[["Material", "Datum"]]
y = df["Menge"]

# Modelltraining
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# Vorhersage für ein bestimmtes Material an einem bestimmten Datum
materialnummer = 90212366  # Beispiel-Materialnummer
datum = pd.to_datetime('2024-01-01').value // 10**9  # Beispiel-Datum
vorhersage = model.predict([[materialnummer, datum]])
print(f"Vorhergesagte Absatzmenge: {vorhersage[0]}")
