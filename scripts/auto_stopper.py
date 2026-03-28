import os
import time
import subprocess

target_file = "checkpoints/checkpoint_epoch_10.pth"

print(f"Monitoring for the creation of {target_file}...")

# Wait until the target file is created
while not os.path.exists(target_file):
    time.sleep(30)  # Check every 30 seconds to save CPU

print("Epoch 10 checkpoint detected! Waiting 10 seconds to ensure the file is completely saved to disk...")
time.sleep(10)

print("Terminating train.py from PowerShell...")
stop_cmd = 'powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq \'python.exe\' -and $_.CommandLine -like \'*scripts/train.py*\' } | Invoke-CimMethod -MethodName Terminate"'
subprocess.run(stop_cmd, shell=True)
print("Training successfully stopped automatically at Epoch 10.")
