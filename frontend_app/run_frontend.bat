@echo off
echo ========================================
echo Traffic Anomaly Detection - Frontend
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call ..\.venv\Scripts\activate.bat

echo.
echo Starting Streamlit frontend...
echo.
echo The application will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit
streamlit run app.py --server.port 8501

pause
