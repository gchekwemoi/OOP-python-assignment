# OOP with Python: Uganda Mini Projects

This repository contains coursework mini projects using
object-oriented Python, statistical analysis and visualisation.

## Progress

- Project 1: District Population Forecaster — implemented.
- Projects 2–5: To be added.

## Repository structure

- project1_population.ipynb: analysis, results, charts and discussion.
- src/population.py: DistrictPopulation class and input validation.
- src/forecasting.py: abstract Forecaster and three forecasting models.
- tests/test_population.py: six unit tests.
- requirements.txt: required Python packages.

## Run in Google Colab — no local installation needed

1. Open project1_population.ipynb on GitHub.
2. Click its Open in Colab badge.
3. Run the notebook from top to bottom.

The first code cell downloads the repository into Colab and
sets the working folder. Colab provides the main analysis packages;
the notebook installs pytest before running the tests.

An internet connection is required. Changes to files in Colab
are not automatically saved back to GitHub.

## Run locally — optional

With Python and Git installed:

```bash
git clone https://github.com/gchekwemoi/OOP-python-assignment.git
cd OOP-python-assignment
python -m pip install -r requirements.txt
python -m notebook
```

Open project1_population.ipynb and choose Restart & Run All.

To run the tests from the repository folder:

```bash
python -m pytest tests/test_population.py -q
```

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

## AI-use declaration

I used ChatGPT to help set up GitHub and Colab, draft Python
classes, analysis code, tests and written explanations, and
troubleshoot file-location and import errors.

I ran the code and tests in Colab and reviewed the outputs.
The additional Jinja and Mbarara population data were
AI-generated synthetic examples.
