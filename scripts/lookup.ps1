# Demo wrapper script for PowerShell
# Allows simple command: lookup diabetes
# Instead of: python main.py "diabetes"

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Query
)

if (-not $Query) {
    Write-Host "Usage: .\lookup <clinical term>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\lookup diabetes"
    Write-Host "  .\lookup blood sugar test"
    Write-Host "  .\lookup metformin 500 mg"
    exit 1
}

# Join all arguments into a single query string
$QueryString = $Query -join " "

# Run the main script
python main.py $QueryString
