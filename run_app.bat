@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Streamlit app...
streamlit run app_improved.py

pause
