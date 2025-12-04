# Batch file to compile the C++ mesh optimizer on Windows
# Run this with: .\compile.bat

# Using MinGW g++
g++ -O3 -std=c++11 mesh_optimizer.cpp -o mesh_optimizer.exe

Write-Host "Compilation complete! mesh_optimizer.exe created." -ForegroundColor Green
