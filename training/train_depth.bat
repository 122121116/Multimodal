@echo off
chcp 65001 >nul

echo ==================================================
echo   LoRA Training - Depth Condition
echo ==================================================

set HF_ENDPOINT=https://hf-mirror.com

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

echo [CHECK] CUDA availability ...
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch not detected, please install it: pip install torch torchvision
    pause
    exit /b 1
)

if not exist "..\dataset_output" (
    echo [ERROR] Dataset directory "..\dataset_output" not found.
    pause
    exit /b 1
)

echo.
echo Starting training ...
echo.

python train_lora.py --condition_type depth --dataset_root ..\dataset_output --num_epochs 25

echo.
echo Training finished.
pause
