# Test embedding API endpoint using EMBED_* env vars (fallback to LLM_*)
# Usage: .\scripts\test_deepseek_embed.ps1

$BaseUrl = if ($env:EMBED_BASE_URL) { $env:EMBED_BASE_URL } else { $env:LLM_BASE_URL }
$ApiKey  = if ($env:EMBED_API_KEY)  { $env:EMBED_API_KEY }  else { $env:LLM_API_KEY }
$Model   = if ($env:EMBED_MODEL)    { $env:EMBED_MODEL }    else { $env:LLM_MODEL }
$Auth    = if ($env:EMBED_AUTH_ENABLED -ne $null) { $env:EMBED_AUTH_ENABLED } else { $env:LLM_AUTH_ENABLED }

if (-not $BaseUrl) { Write-Host 'ERROR: EMBED_BASE_URL (or LLM_BASE_URL) not set' -ForegroundColor Red; exit 1 }
if (-not $Model)   { Write-Host 'ERROR: EMBED_MODEL (or LLM_MODEL) not set'       -ForegroundColor Red; exit 1 }

# Auth: if explicitly disabled, skip the key check
$NeedAuth = ($Auth -ne 'false')
if ($NeedAuth -and -not $ApiKey) {
    Write-Host 'ERROR: EMBED_API_KEY (or LLM_API_KEY) not set, and auth is enabled' -ForegroundColor Red
    exit 1
}

# Build the embeddings URL
$Base = $BaseUrl -replace '/v1/?$', ''
$EmbedUrl = "$Base/v1/embeddings"

$Body = @{
    model = $Model
    input = 'hello test embedding'
} | ConvertTo-Json

Write-Host '=== Test Embedding API ===' -ForegroundColor Cyan
Write-Host "URL:          $EmbedUrl"
Write-Host "Model:        $Model"
Write-Host "Auth enabled: $Auth"
if ($NeedAuth) { Write-Host "API Key:      <set, length=$($ApiKey.Length)>" }
Write-Host ''

try {
    $Headers = @{ 'Content-Type' = 'application/json' }
    if ($NeedAuth) { $Headers['Authorization'] = "Bearer $ApiKey" }

    $Response = Invoke-RestMethod -Uri $EmbedUrl `
        -Method Post `
        -Headers $Headers `
        -Body $Body

    Write-Host 'SUCCESS (HTTP 200)' -ForegroundColor Green
    Write-Host ''

    if ($Response.data -and $Response.data[0].embedding) {
        $emb = $Response.data[0].embedding
        Write-Host "Embedding dimension: $($emb.Count)"
        Write-Host "First 5 values:      [$([string]::Join(', ', $emb[0..4])) ...]"
        Write-Host ''
        Write-Host 'OK - Embedding API works!' -ForegroundColor Green
    } else {
        Write-Host 'WARNING: Response received but no data[0].embedding field' -ForegroundColor Yellow
        $Response | ConvertTo-Json -Depth 1
    }
} catch [System.Net.WebException] {
    $Resp = $_.Exception.Response
    if ($Resp) {
        $StatusCode = [int]$Resp.StatusCode
        Write-Host "FAILED (HTTP $StatusCode)" -ForegroundColor Red

        if ($StatusCode -eq 404) {
            Write-Host ''
            Write-Host 'The embedding endpoint /v1/embeddings returned 404.' -ForegroundColor Red
            Write-Host 'This provider may not support embeddings, or the URL is wrong.'
        } elseif ($StatusCode -eq 401 -or $StatusCode -eq 403) {
            Write-Host ''
            Write-Host 'Authentication failed. Check EMBED_API_KEY and EMBED_AUTH_ENABLED.' -ForegroundColor Yellow
        }
    } else {
        Write-Host "NETWORK ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
} catch {
    Write-Host "NETWORK ERROR: $_" -ForegroundColor Red
}
