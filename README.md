# OOP with Python: Uganda Mini-Projects

Five mini-projects for  Object-Oriented Programming with Python. The notebooks demonstrate reusable Python classes, validation, statistics, numerical methods, forecasting, visualisation and tests in Ugandan scenarios. Data are illustrative unless explicitly identified as NASA POWER observations.

## Progress and notebooks

| Project | Notebook | Focus |
| --- | --- | --- |
| 1. UBOS District Population Forecaster | [project1_population.ipynb](project1_population.ipynb) | Linear, CAGR and Fibonacci forecasts; classroom planning |
| 2. Solar Micro-Grid Dispatch Planner | [project2_microgrid.ipynb](project2_microgrid.ipynb) | Solar/battery allocation, infeasibility, costs and hybrid extension |
| 3. Lake Victoria Fish Stock & Export Risk | [project3_fish_risk.ipynb](project3_fish_risk.ipynb) | Logistic fish stock, harvest scenarios and revenue risk |
| 4. Rainfall Pattern & Crop Suitability | [project4_rainfall.ipynb](project4_rainfall.ipynb) | Regional rainfall, crop screening and 10-year real-data extension |
| 5. Taxi Route Revenue, Pricing & Fleet Planner | [project5_taxi_planner.ipynb](project5_taxi_planner.ipynb) | Fare equilibrium, walk-forward forecasts and vehicle allocation |

**All five notebook files have been saved.** For reproducibility, each notebook should be run from start to finish in a fresh Python environment with the required dependencies installed. Local Jupyter execution may require adapting Colab-specific setup cells; successful execution in Anaconda has not yet been independently verified.

## Repository structure

- `src/population.py`, `src/forecasting.py`: district data and population forecasters.
- `src/microgrid.py`, `src/hybrid_microgrid.py`: solar/battery and diesel-extended energy models.
- `src/fish_stock.py`, `src/price_model.py`, `src/risk_assessor.py`: fish stock, prices and revenue risk.
- `src/rainfall.py`, `src/crop_rule.py`, `src/rainfall_analysis.py`: rainfall, crop screening and seasonal analysis.
- `src/taxi_route.py`, `src/taxi_forecasting.py`: taxi route revenue and demand forecasting.
- `tests/test_population.py`, `tests/test_microgrid.py`, `tests/test_fish_model.py`, `tests/test_rainfall.py`, `tests/test_taxi.py`: project-specific pytest tests.
- `requirements.txt`: Python dependencies.
- `microgrid_30days.csv` and `uganda_real_rainfall_2015_2024.csv` are generated in the Colab working folder by the corresponding notebook cells; they need not be pre-existing repository files.

## Setup and execution

### Anaconda and Jupyter Notebook (recommended)

1. Install Anaconda (Python distribution), which includes Jupyter Notebook and a Python environment manager.
2. Download this repository as a ZIP from GitHub and extract it, or clone it using Git.
3. Open **Anaconda Prompt**, navigate to the extracted project folder, and install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open any of the five `.ipynb` files and run its cells in order. Keep the `src/` and `tests/` directories alongside the notebooks.

Some notebook setup cells use Google Colab commands such as `!git clone` and `%cd /content/...`. Those cells should be skipped or adapted to the local project folder when using Anaconda/Jupyter. Interactive cells may request user input.

### Google Colab

The notebooks were developed and tested using Google Colab. To reproduce them there, open a notebook from GitHub and run the cells; its setup code retrieves the project modules. Save modifications back to GitHub separately.

The NASA POWER rainfall extension requires internet access in either environment.

## Tests

From the repository root:

```bash
python -m pytest tests/ -v
```

The individual notebooks show saved successful test runs for Projects 1, 3, 4 and 5; the Project 2 tests are included in `tests/test_microgrid.py`. A combined fresh test run is recommended before final submission.

## Methods, findings and limitations

## Project 1: Methods and findings

Population estimates cover 2015–2024 and are measured in thousands.
Kampala, Wakiso and Gulu use illustrative assignment data.
Jinja and Mbarara use additional synthetic data.

Linear, CAGR and Fibonacci-ratio models were trained on
2015–2021 and evaluated on 2022–2024 using MAE, RMSE and MAPE.
CAGR had the lowest values for all three measures in every district.
The selected models were then refitted using all ten years.

Wakiso had the highest historical CAGR, at 6.47%.
Its forecast population reaches approximately 2.285 million
in 2029, compared with Kampala's 2.255 million.

Assuming 18% of residents are of primary-school age and
53 pupils per classroom, population growth from 2024 to 2029
implies the following additional classroom requirements:

| District | Additional classrooms |
|----------|----------------------:|
| Kampala | 1,545 |
| Wakiso | 2,088 |
| Gulu | 412 |
| Jinja | 200 |
| Mbarara | 275 |

These estimates assume existing capacity meets 2024 needs.

The extension uses 1,000 bootstrap paths with random seed 42
to estimate approximate 95% prediction intervals by resampling
centred annual log-growth residuals.

## Limitations

The data are synthetic and are not official UBOS estimates.
The evaluation period contains only three years.
CAGR assumes constant percentage growth.
Bootstrap intervals exclude parameter uncertainty and structural
changes. Classroom estimates exclude existing shortages and
differences in enrolment and local capacity.

## Validation

Six unit tests passed. They cover valid data, negative populations,
unequal input lengths, empty input, and hand-checkable linear
and CAGR forecasts.

The notebook was also checked using Colab's Restart and run all.

### Project 2: Solar Micro-Grid Dispatch Planner

A `MicroGrid` class solves the assignment's two-demand, two-source linear system for solar and battery allocation. A seeded 30-day synthetic demand series with weekly variation is saved as CSV. The notebook compares looped versus vectorised `scipy.linalg.solve`, flags negative allocations and demonstrates non-negative least squares (NNLS); it computes usage statistics and costs at the illustrative UGX 150/kWh solar and UGX 450/kWh battery rates. For the sample D1=80, D2=90, the solution is 20 kWh solar and 10 kWh battery. Extensions cover a `HybridMicroGrid` with diesel and 1,000 Monte Carlo demand perturbations (±5%). NNLS may leave unmet demand; this simplified system does not model capacity, charging efficiency or the true dispatch conditions of a health centre. Six pytest tests are provided.

### Project 3: Lake Victoria Fish Stock & Export Risk

A `FishStock` class applies logistic growth with harvesting for 52 weeks (initial stock 4,000 tonnes, r=0.4, K=10,000). `PriceModel` generates seeded bounded fish-price paths and estimates revenue; `RiskAssessor` provides descriptive measures, CV-based illustrative risk bands and the lower 5th percentile of simulated annual revenue. Harvesting rates of 5%, 10%, 20% and 30% are compared with the theoretical maximum sustainable yield (rK/4 = 1,000 tonnes per week). The notebook uses 1,000 price paths and compares an eight-week annual closure with uninterrupted harvesting over five years. Prices and biological parameters are illustrative; the price-based risk simulation holds harvesting quantities fixed. Ten pytest tests passed in the saved notebook.

### Project 4: Rainfall Pattern & Crop Suitability Analyser

`Region`, `CropRule` and `RainfallAnalyser` model monthly rainfall, crop screening thresholds, cosine similarity, Pearson correlation, Euclidean distance and peak-based rainy-season patterns for Kampala, Gulu and Mbarara. The original illustrative dataset produces annual totals of 1,600, 1,388 and 1,040 mm respectively. The extension downloads **2015–2024 monthly precipitation** for three selected city coordinates from **NASA POWER** (`PRECTOTCORR_SUM`), creates per-month box plots of interannual variability and a second suitability heatmap based on ten-year monthly means. Crop monthly thresholds are simplified modelling assumptions, not validated agronomic planting recommendations. The saved notebook shows ten pytest tests passed.

Source: NASA Langley Research Center, Prediction of Worldwide Energy Resources (POWER) project — https://power.larc.nasa.gov/ . Downloaded parameter: `PRECTOTCORR_SUM` (2015–2024).

### Project 5: Taxi Route Revenue, Pricing & Fleet Planner

A `Route` class stores daily passenger counts and fares for Kampala–Ntinda, Kampala–Entebbe and Kampala–Mukono. An abstract `TaxiForecaster` supports a three-day moving average, exponential smoothing and linear trend. The notebook solves the illustrative Ntinda supply-demand equilibrium using SciPy (UGX 2,200 equilibrium fare versus UGX 2,000 current fare), tunes alpha and backtests Days 4–10 using MAE, forecasts Day 11 demand/revenue, and estimates vehicles with a 15% buffer. **Each taxi seats at most 14 passengers per trip and is assumed to make eight one-way trips daily**, so 112 is its maximum *daily passenger journeys*, never simultaneous occupancy. Under these assumptions, the saved notebook estimates three vehicles in total, one per route. A 60-day seeded weekly-seasonal simulation finds the seasonal-naïve method better than the three-day moving average. Actual peak demand, traffic and real fleet operations are excluded. Ten pytest tests passed in the saved notebook.

## Academic integrity and AI-use declaration

I used Ai as a learning and coding assistant to help with
GitHub and Colab setup, draft code and explanations, and
troubleshoot errors. AI assistance included suggestions for
classes, forecasting methods, tests and synthetic data for
Jinja and Mbarara.

I used my own judgement to review the suggestions, examine
the results and consider whether the assumptions were
reasonable. I ran the code and tests, compared model errors,
and reviewed the population forecasts and classroom estimates.

The additional district data are synthetic, not official
population estimates.
