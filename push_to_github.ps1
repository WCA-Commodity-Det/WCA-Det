$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GitExe = "D:\Git\cmd\git.exe"
$RemoteUrl = "https://github.com/WCA-Commodity-Det/WCA-Det.git"

Set-Location $RepoDir

if (-not (Test-Path -LiteralPath $GitExe)) {
    throw "Git was not found at $GitExe"
}

if ((Test-Path -LiteralPath ".git") -and -not (Test-Path -LiteralPath ".git\config")) {
    Write-Host "Removing incomplete .git directory from a failed initialization..."
    Remove-Item -LiteralPath ".git" -Recurse -Force
}

if (-not (Test-Path -LiteralPath ".git")) {
    & $GitExe init
    & $GitExe branch -M main
}

& $GitExe config user.name "Jinliang Zhang"
& $GitExe config user.email "2025720781@yangtzeu.edu.cn"

$Remotes = (& $GitExe remote) -join "`n"
if ($Remotes -match "(^|`n)origin($|`n)") {
    & $GitExe remote set-url origin $RemoteUrl
}
else {
    & $GitExe remote add origin $RemoteUrl
}

& $GitExe add .
& $GitExe status --short
& $GitExe commit -m "Release WCA-Det code and commodity dataset"
& $GitExe push -u origin main

