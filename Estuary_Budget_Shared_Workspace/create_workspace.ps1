$Root = Join-Path $PWD "Estuary_Budget_Shared_Workspace"
$Folders = @("Spoon_Model", "Pioneer_Model", "Budget_Analysis", "Documents", "In_Progress")
New-Item -ItemType Directory -Force -Path $Root | Out-Null
foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $Folder) | Out-Null
}
Write-Host "Workspace created at: $Root"
