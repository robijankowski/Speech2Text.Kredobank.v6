@echo off
REM Check if the virtual environment directory already exists
IF NOT EXIST "venv" (
    REM Create the virtual environment
    python -m venv venv
    echo Virtual environment created.
) ELSE (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependencies from requirements.txt
pip install -r requirements.txt

REM Deactivate the virtual environment
deactivate

echo Virtual environment setup complete and dependencies installed.
pause
