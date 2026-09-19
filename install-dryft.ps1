$ErrorActionPreference = "Stop"

$Version = if ($env:DRYFT_CLI_VERSION) { $env:DRYFT_CLI_VERSION } else { "0.1.0" }
$InstallDir = if ($env:DRYFT_CLI_INSTALL_DIR) {
    $env:DRYFT_CLI_INSTALL_DIR
} else {
    Join-Path $PSScriptRoot "bin"
}

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    throw "dryft: this installer is for Windows; use install-dryft.sh on macOS or Linux"
}
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "X64") {
    throw "dryft: Windows binaries are currently published for x86_64 only"
}

$Asset = "dryft-windows-x86_64.zip"
$BaseUrl = if ($env:DRYFT_CLI_BASE_URL) {
    $env:DRYFT_CLI_BASE_URL.TrimEnd("/")
} else {
    "https://dryft-ai.github.io/kernel-deployment-htn-2026/releases/dryft-cli-v$Version"
}
$IsHttps = $BaseUrl.StartsWith("https://", [System.StringComparison]::OrdinalIgnoreCase)
$IsAllowedHttp = $BaseUrl.StartsWith("http://", [System.StringComparison]::OrdinalIgnoreCase) `
    -and $env:DRYFT_CLI_ALLOW_INSECURE -eq "1"
if (-not ($IsHttps -or $IsAllowedHttp)) {
    throw "dryft: release URL must use HTTPS"
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("dryft-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $TempDir | Out-Null
try {
    $Archive = Join-Path $TempDir $Asset
    $ChecksumFile = "$Archive.sha256"
    Write-Host "Downloading dryft $Version for windows-x86_64..."
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$Asset" -OutFile $Archive
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$Asset.sha256" -OutFile $ChecksumFile

    $Expected = ((Get-Content -Raw $ChecksumFile).Trim() -split '\s+')[0].ToLowerInvariant()
    if ($Expected -notmatch '^[0-9a-f]{64}$') {
        throw "dryft: release checksum is invalid"
    }
    $Actual = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "dryft: checksum verification failed"
    }

    $Unpacked = Join-Path $TempDir "unpacked"
    Expand-Archive -Path $Archive -DestinationPath $Unpacked
    $Executable = Join-Path $Unpacked "dryft.exe"
    if (-not (Test-Path -PathType Leaf $Executable)) {
        throw "dryft: release archive does not contain dryft.exe"
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $Destination = Join-Path $InstallDir "dryft.exe"
    Copy-Item -Force $Executable $Destination
    & $Destination version
    Write-Host "Installed $Destination"
} finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
