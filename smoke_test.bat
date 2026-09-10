@echo off
REM Double-click this FIRST. It takes about two minutes and checks that
REM everything works before you start the real run.
cd /d "%~dp0"
call "%~dp0run.bat" --smoke
