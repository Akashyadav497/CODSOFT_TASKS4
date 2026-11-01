@echo off
REM Sales Prediction Windows Quick Run Script

set PYTHONPATH=.

if "%1"=="" goto help
if "%1"=="install" goto install
if "%1"=="eda" goto eda
if "%1"=="train" goto train
if "%1"=="evaluate" goto evaluate
if "%1"=="predict" goto predict
if "%1"=="test" goto test
if "%1"=="app" goto app

:help
echo Sales Prediction CLI Helper
echo Usage: run.bat [command]
echo.
echo Commands:
echo   install   - Install dependencies
echo   eda       - Run EDA and generate figures
echo   train     - Train models and tune hyperparameters
echo   evaluate  - Evaluate test metrics and generate diagnostic plots
echo   predict   - Run sample prediction CLI
echo   test      - Run pytest automated tests
echo   app       - Launch Streamlit dashboard
goto end

:install
pip install -r requirements.txt
goto end

:eda
python -m src.eda
goto end

:train
python -m src.train
goto end

:evaluate
python -m src.evaluate
goto end

:predict
python -m src.predict --tv 150 --radio 30 --newspaper 20
goto end

:test
pytest tests/ -v
goto end

:app
streamlit run app.py
goto end

:end
