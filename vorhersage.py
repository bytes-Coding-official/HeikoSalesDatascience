# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# 
# # Annahme: dataframe enthält historische Daten
# dataframe = pd.read_excel("file.xlsx", sheet_name="Sheet1")
# dataframe.columns = ['Material', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', "total"]
# 
# # Filtern nach spezifischem Material
# df = dataframe.loc[dataframe['Material'] == "Anti Claro/Neptra"]
# 
# # Aufteilen in Features und Zielvariable
# X = df.iloc[:, 1:13]  # Monatliche Verkaufszahlen
# y = df.iloc[:, 13]    # Gesamtverkaufszahl (Summe der Monatszahlen)
# 
# # Modelltraining
# model = LinearRegression()
# model.fit(X, y)
# 
# # Vorhersage der Gesamtverkaufszahl
# y_pred = model.predict(X)
# 
# # Ausgabe der Vorhersage und der tatsächlichen Gesamtverkaufszahl
# print("Vorhergesagte Gesamtverkaufszahl: ", y_pred)
# print("Tatsächliche Gesamtverkaufszahl: ", y)
# 
# # Leistungsbewertung (in diesem Fall ist es eine Form der Überprüfung)
# print("MSE: ", mean_squared_error(y, y_pred))
# 

from pmdarima import auto_arima
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Daten laden
dataframe = pd.read_excel("file.xlsx", sheet_name="Sheet1")
dataframe.columns = ['Material', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', "total"]

# Daten für ein spezifisches Material auswählen
df = dataframe.loc[dataframe['Material'] == "Anti Claro/Neptra"]

# Monatliche Verkaufszahlen in eine Zeitreihe umwandeln
sales_data = df.iloc[0, 1:13].astype(float)

# ARIMA-Modell erstellen und anpassen
model = ARIMA(sales_data, order=(1, 1, 1))
fitted_model = model.fit()

# Vorhersage für die nächsten 12 Monate
forecast = fitted_model.forecast(steps=12)

# Vorhersage anzeigen
print(forecast)

#füge 13 weitere columns an das df an
df = df.reindex(df.columns.tolist() + list(range(13, 25)), axis=1)
#add forecast to df
df.iloc[0, 13:25] = forecast
print(df.head())

#save df to excel
df.to_excel('output.xlsx', index=False)



# Plot (optional)
plt.figure(figsize=(10, 6))
plt.plot(sales_data, label='Historische Verkaufszahlen')
plt.plot(np.arange(len(sales_data), len(sales_data) + 12), forecast, label='Vorhersage')
plt.legend()
plt.show()
