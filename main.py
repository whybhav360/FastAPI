import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# model intigration
df = pd.read_csv('Processed.csv')
df.columns = df.columns.str.strip()  # Clean column names
df = df.dropna(subset=['year', 'country', 'apricot'])  # Remove incomplete rows
df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
df = df.dropna(subset=['year'])  # Drop invalid dates
df = df.set_index('year').sort_index()

country_series = df.pivot(columns='country', values='apricot')

# Forecasting
def forecast_country(series, country_name, forecast_years=5):
    try:
        series.index = pd.to_datetime(series.index, errors='coerce')
        series = series.asfreq('YS')  # Set frequency to Year Start
        series = series.dropna().sort_index()
        series = series.interpolate().ffill().bfill()

        #  ARIMA 
        model = auto_arima(series, seasonal=False, trace=False, suppress_warnings=True)
        final_model = ARIMA(series, order=model.order)
        model_fit = final_model.fit()

        # model fitting
        forecast = model_fit.get_forecast(steps=forecast_years)

        last_year = series.index[-1].year
        future_years = [last_year + i for i in range(1, forecast_years + 1)]

        return pd.DataFrame({
            'country': country_name,
            'year': future_years,
            'predicted_apricot_growth': forecast.predicted_mean.values,
            'lower_ci': forecast.conf_int().iloc[:, 0].values,
            'upper_ci': forecast.conf_int().iloc[:, 1].values
        })
    except Exception as e:
        print(f"[ERROR] Forecast failed for {country_name}: {e}")
        return None

# Input
class CountryRequest(BaseModel):
    country: str
    forecast_years: int = 5

@app.post("/predict/")
async def get_forecast(request: CountryRequest):
    country_name = request.country
    forecast_years = request.forecast_years

    if country_name not in country_series.columns:
        raise HTTPException(status_code=404, detail=f"Country {country_name} not found in dataset.")

    country_data = country_series[country_name].dropna()
    if len(country_data) <= 5:  
        raise HTTPException(status_code=400, detail="Not enough data to forecast for this country.")
    
    forecast_df = forecast_country(country_data, country_name, forecast_years)
    if forecast_df is None:
        raise HTTPException(status_code=400, detail="Error generating forecast.")
    
    historical_data = country_series[country_name].last('5Y')
    historical_data.index = pd.to_datetime(historical_data.index)
    
    all_years = list(historical_data.index.year) + list(forecast_df['year'])
    all_values = list(historical_data.values) + list(forecast_df['predicted_apricot_growth'])

    plt.figure(figsize=(8, 6))
    plt.plot(all_years, all_values, label='Historical + Forecast', marker='o', color='blue')
    plt.plot(forecast_df['year'], forecast_df['predicted_apricot_growth'],
             label='Forecast (Next 5 Years)', marker='o', color='orange')
    plt.fill_between(forecast_df['year'],
                     forecast_df['lower_ci'],
                     forecast_df['upper_ci'],
                     color='orange', alpha=0.2)

    plt.title(f'Apricot Production Forecast - {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Apricot Production')
    plt.legend()
    plt.grid(True)
    plt.xticks(list(historical_data.index.year) + list(forecast_df['year']), rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    result = {
        'forecast_data': forecast_df.to_dict(orient='records'),
        'historical_data': {
            'years': list(historical_data.index.year),
            'values': historical_data.values.tolist()
        },
        'image': f"data:image/png;base64,{img_base64}"
    }

    return JSONResponse(content=result)
