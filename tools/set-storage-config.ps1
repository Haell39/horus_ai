param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('local','onedrive')]
    [string]$Mode = 'local',

    [Parameter(Mandatory=$false)]
    [string]$LocalPath = '',

    [Parameter(Mandatory=$false)]
    [string]$OneDriveLink = '',

    [Parameter(Mandatory=$false)]
    [string]$ServerUrl = 'http://localhost:8000'
)

# Small helper to POST a storage config to the backend admin endpoint
try {
    $body = @{ mode = $Mode }
    if ($Mode -eq 'local' -and $LocalPath) { $body.local_path = $LocalPath }
    if ($Mode -eq 'onedrive' -and $OneDriveLink) { $body.oneDriveLink = $OneDriveLink }

    $json = $body | ConvertTo-Json -Depth 4

    $uri = "$ServerUrl/api/v1/admin/storage-config"
    Write-Host "Posting storage config to $uri" -ForegroundColor Cyan
    Write-Host "Payload: $json" -ForegroundColor DarkGray

    $resp = Invoke-RestMethod -Uri $uri -Method Post -Body $json -ContentType 'application/json'
    Write-Host "Server response:" -ForegroundColor Green
    $resp | ConvertTo-Json -Depth 4 | Write-Output
} catch {
    Write-Host "Failed to set storage config: $_" -ForegroundColor Red
    exit 1
}
