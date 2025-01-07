@echo off
:: Request the first two variables from the user
set /p userid=Enter the participant ID: 
set /p cond=Enter the study condition (1: control, 2: card, 3: in situ): 

set round_num=0

:: Request another variable from the user after the script ends
set /p new_round_num=Enter starting round (ENTER to use round %round_num%):

set /p log_data=Log data? (y/n):

if "%cond%"=="1" set cond="control"
if "%cond%"=="2" set cond="card"
if "%cond%"=="3" set cond="in situ"
if "%new_round_num%"=="" (goto start) else (set round_num=%new_round_num%)

:start
call "C:\Users\Ryan\PycharmProjects\maisr\.venv\Scripts\activate.bat"
:: Run the Python script and pass the variables to it
echo Starting pygame with ID %userid%, condition %cond% and round %round_num%
python main.py %userid% %cond% %round_num% %log_data%
goto end

:end
echo Exiting script...
set /p nothing=Press enter to continue
