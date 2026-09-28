<#
Windows setup for the Healthcare Anomaly and Outbreak Detection project
(academic demonstration with SYNTHETIC data only).

What it does:
  1. Checks that Java 17 is available.
  2. Checks that the project virtual environment (.venv) exists.
  3. Downloads one small pure-Java JAR from Maven Central into jars\ and
     verifies its checksums. This JAR lets Spark use local files on Windows
     without winutils.exe or hadoop.dll.

What it does NOT do:
  It does not need administrator access and does not change PATH, JAVA_HOME,
  the PowerShell execution policy, Windows services, firewall or network settings.

Run from the project folder:
  .\setup_windows.ps1
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # much faster downloads in Windows PowerShell 5.1

$ProjectRoot = $PSScriptRoot
$JarDir = Join-Path $ProjectRoot "jars"
$JarPath = Join-Path $JarDir "hadoop-bare-naked-local-fs-0.1.0.jar"
$JarUrl = "https://repo1.maven.org/maven2/com/globalmentor/hadoop-bare-naked-local-fs/0.1.0/hadoop-bare-naked-local-fs-0.1.0.jar"

# SHA-1 published by Maven Central in <JarUrl>.sha1
# (Maven Central publishes no SHA-256 or SHA-512 file for this artifact.)
$ExpectedSha1 = "cd03dc0f6e2b8d8957d97d421e95d9ceaa16b06b"
# SHA-256 of the same JAR, computed after it matched the official SHA-1 above.
$ExpectedSha256 = "e0cc30fb0531eb0b59468dc0abf5b257533d2365b5e9f45e795edd707aa78c62"

function Stop-Setup([string]$Message) {
    Write-Host ""
    Write-Host "SETUP FAILED: $Message" -ForegroundColor Red
    exit 1
}

function Test-JarChecksums([string]$Path) {
    $sha256 = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
    $sha1 = (Get-FileHash -Path $Path -Algorithm SHA1).Hash.ToLower()
    return ($sha256 -eq $ExpectedSha256) -and ($sha1 -eq $ExpectedSha1)
}

Write-Host "=== Windows setup: Healthcare Anomaly and Outbreak Detection (synthetic data) ==="

# ---------------------------------------------------------------- 1. Java 17
Write-Host ""
Write-Host "[1/3] Checking Java 17..."
$JavaExe = $null
if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    $JavaExe = Join-Path $env:JAVA_HOME "bin\java.exe"
} elseif (Get-Command java -ErrorAction SilentlyContinue) {
    $JavaExe = (Get-Command java).Source
}
if (-not $JavaExe) {
    Stop-Setup ("Java was not found. Install Java 17, then open a NEW PowerShell window and run this script again:`n" +
                "  winget install --id EclipseAdoptium.Temurin.17.JDK --exact")
}
# 'java -version' prints to stderr, so run it through cmd to capture the text safely.
$JavaVersionText = cmd /c "`"$JavaExe`" -version 2>&1" | Out-String
if ($JavaVersionText -notmatch 'version "(\d+)') {
    Stop-Setup "Could not read the Java version from: $JavaExe"
}
if ([int]$Matches[1] -ne 17) {
    Stop-Setup ("Java $($Matches[1]) was found at $JavaExe, but this project was tested with Java 17.`n" +
                "Install it with: winget install --id EclipseAdoptium.Temurin.17.JDK --exact")
}
Write-Host "  OK: Java 17 found ($JavaExe)"

# ---------------------------------------------------------------- 2. Virtual environment
Write-Host ""
Write-Host "[2/3] Checking the project virtual environment..."
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Stop-Setup (".venv was not found. Create it and install the packages first:`n" +
                "  py -3.12 -m venv .venv`n" +
                "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt")
}
Write-Host "  OK: $VenvPython"

# ---------------------------------------------------------------- 3. Compatibility JAR
Write-Host ""
Write-Host "[3/3] Checking the Spark Windows compatibility JAR..."
if (-not (Test-Path $JarDir)) {
    New-Item -ItemType Directory -Path $JarDir | Out-Null
}

if ((Test-Path $JarPath) -and (Test-JarChecksums $JarPath)) {
    Write-Host "  OK: verified JAR already present, download skipped ($JarPath)"
} else {
    if (Test-Path $JarPath) {
        Write-Host "  Existing JAR failed checksum verification; downloading it again."
        Remove-Item $JarPath
    }
    Write-Host "  Downloading $JarUrl"
    $DownloadPath = "$JarPath.download"
    try {
        # Allow TLS 1.2 for this PowerShell process only (needed by older Windows PowerShell).
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $JarUrl -OutFile $DownloadPath -UseBasicParsing
    } catch {
        if (Test-Path $DownloadPath) { Remove-Item $DownloadPath }
        Stop-Setup "Download failed: $($_.Exception.Message)"
    }
    if (-not (Test-JarChecksums $DownloadPath)) {
        Remove-Item $DownloadPath
        Stop-Setup ("The downloaded JAR does not match the expected checksums and was deleted.`n" +
                    "  expected SHA-256: $ExpectedSha256`n" +
                    "  expected SHA-1:   $ExpectedSha1")
    }
    Move-Item -Path $DownloadPath -Destination $JarPath
    Write-Host "  OK: downloaded and verified ($JarPath)"
}
Write-Host "  SHA-256: $ExpectedSha256"

# ---------------------------------------------------------------- Next steps
Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m pytest"
Write-Host "  .\.venv\Scripts\python.exe generate_data.py --mode train"
Write-Host "  .\.venv\Scripts\python.exe generate_data.py --mode stream --minutes 2 --fresh"
