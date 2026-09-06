@echo off
chcp 65001 >nul
title DETR full training (runs/full)
cd /d E:\Renming\experiments\detr
powershell -NoProfile -Command "python -u train.py --epochs 40 --w-layout 0.5 --w-sym 0.5 --out runs/full --resume 2>&1 | Tee-Object -FilePath 'E:\Renming\experiments\detr\runs\full_train.log'"
pause
