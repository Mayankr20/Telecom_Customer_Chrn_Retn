@echo off
title Telco Customer Churn & Retention Command Center
cd /d "%~dp0"
echo ====================================================================
echo      🎯 TELCO CUSTOMER CHURN & RETENTION COMMAND CENTER
echo ====================================================================
echo.
echo Choose your preferred launch mode:
echo   [1] Launch Modern Streamlit Web App (app.py) - Full Machine Learning
echo   [2] Open Standalone Instant Web Dashboard (dashboard.html) - Zero Install
echo   [3] Retrain / Re-fit ML Pipeline (train_model.py)
echo.
set /p choice="Enter your choice (1, 2, or 3, default is 1): "
if "%choice%"=="2" (
    echo.
    echo Opening dashboard.html in your default web browser...
    start "" dashboard.html
) else if "%choice%"=="3" (
    echo.
    echo Retraining model pipeline...
    python train_model.py
    pause
) else (
    echo.
    echo Starting Streamlit server...
    python -m streamlit run app.py
)
pause
