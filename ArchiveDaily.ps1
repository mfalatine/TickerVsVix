# ArchiveDaily.ps1 — One-click daily archive of CSVs to C:\ProgramData\Stratagem571\VixLogs\YYMMDD
# Usage (desktop shortcut target):
#   powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\ArchiveDaily.ps1"
# Optional: powershell.exe -ExecutionPolicy Bypass -File "...\ArchiveDaily.ps1" -Date 2025-08-20

param(
  [string]$Date
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Read Netlify site base from file (contains fully qualified https URL), or set it here directly.
$siteFile = Join-Path $PSScriptRoot 'netlify.app.location.txt'
if (Test-Path $siteFile) {
  $SiteBase = (Get-Content $siteFile -Raw).Trim()
}
if (-not $SiteBase) {
  throw "Missing Netlify site base. Put your site URL in netlify.app.location.txt (e.g., https://tickervsvix.netlify.app)."
}

function Get-ETNowParts {
  # Returns a hashtable with ET date and minutes from midnight
  $etTz = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
  $nowUtc = [DateTime]::UtcNow
  $nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc($nowUtc, $etTz)
  $date = $nowEt.ToString('yyyy-MM-dd')
  $minutes = [int]$nowEt.TimeOfDay.TotalMinutes
  return @{ Date = $date; Minutes = $minutes }
}

function Resolve-TargetDate {
  param([string]$UserDate)
  if ($UserDate) { return $UserDate }
  $p = Get-ETNowParts
  if ($p.Minutes -lt (16*60 + 15)) {
    # before 16:15 ET → yesterday
    $d = [DateTime]::ParseExact($p.Date,'yyyy-MM-dd',$null).AddDays(-1)
  } else {
    $d = [DateTime]::ParseExact($p.Date,'yyyy-MM-dd',$null)
  }
  return $d.ToString('yyyy-MM-dd')
}

function Get-Interval {
  param([string]$TargetDate)
  $etTz = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
  $nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $etTz)
  $d = [DateTime]::ParseExact($TargetDate,'yyyy-MM-dd',$null)
  $ago = ($nowEt.Date - $d.Date).TotalDays
  if ($ago -le 25) { return '1m' } else { return '5m' }
}

function Invoke-YahooChart {
  param(
    [string]$Symbol,
    [string]$TargetDate,
    [string]$Interval
  )
  # Use range window; we will filter to session ET
  $range = if ($Interval -eq '1m') { '5d' } else { '1mo' }
  $uri = "$SiteBase/.netlify/functions/fetch_chart?symbol=$([uri]::EscapeDataString($Symbol))&interval=$Interval&range=$range&includePrePost=false"
  return Invoke-RestMethod -Method GET -Uri $uri -Headers @{ 'Cache-Control' = 'no-store' }
}

function Convert-ToEtDatetime {
  param([long[]]$UnixSeconds)
  $etTz = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
  $dates = @()
  foreach ($s in $UnixSeconds) {
    $utc = [DateTimeOffset]::FromUnixTimeSeconds($s).UtcDateTime
    $et  = [System.TimeZoneInfo]::ConvertTimeFromUtc($utc, $etTz)
    $dates += $et
  }
  return ,$dates
}

function Build-SessionRows {
  param(
    [DateTime[]]$TsEt,
    [double[]]$Close,
    [string]$TargetDate
  )
  # Return map: 'yyyy-MM-dd HH:mm' -> value (only 09:30–16:15 ET on TargetDate)
  $rows = @{}
  for ($i=0; $i -lt $TsEt.Count; $i++) {
    $t = $TsEt[$i]
    if ($t.ToString('yyyy-MM-dd') -ne $TargetDate) { continue }
    $hm = $t.ToString('HH:mm')
    if ($hm -lt '09:30' -or $hm -gt '16:15') { continue }
    $k = $t.ToString('yyyy-MM-dd HH:mm')
    $rows[$k] = [double]$Close[$i]
  }
  return $rows
}

function Write-CsvPair {
  param(
    [hashtable]$TickerRows,
    [hashtable]$VixRows,
    [string]$Label,
    [string]$TargetDate,
    [string]$OutDir
  )
  $keys = @($TickerRows.Keys | Where-Object { $VixRows.ContainsKey($_) } | Sort-Object)
  if (-not $keys) { return }
  $outPath = Join-Path $OutDir ("{0}_vix_1min_{1}_0930_1615_ET.csv" -f $Label.ToLower(), $TargetDate)
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("timestamp_et,ticker_close,vix_close")
  foreach ($k in $keys) {
    $line = "{0},{1},{2}" -f $k, ([double]$TickerRows[$k]).ToString(), ([double]$VixRows[$k]).ToString()
    [void]$sb.AppendLine($line)
  }
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  [IO.File]::WriteAllText($outPath, $sb.ToString())
}

$targetDate = Resolve-TargetDate -UserDate $Date
$interval = Get-Interval -TargetDate $targetDate

$labels = 'ES','MES','NQ','MNQ','YM','RTY'
$yahoo = @{ ES='ES=F'; MES='MES=F'; NQ='NQ=F'; MNQ='MNQ=F'; YM='YM=F'; RTY='RTY=F' }

$outRoot = 'C:\ProgramData\Stratagem571\VixLogs'
$outDir  = Join-Path $outRoot ((Get-Date $targetDate).ToString('yyMMdd'))

Write-Host ("Archiving {0} tickers for {1} to {2}" -f $labels.Count, $targetDate, $outDir)

# Overwrite existing YYMMDD folder if present
if (Test-Path $outDir) {
  try { Remove-Item -Recurse -Force $outDir } catch {}
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($label in $labels) {
  $sym = $yahoo[$label]
  $respT = Invoke-YahooChart -Symbol $sym   -TargetDate $targetDate -Interval $interval
  $respV = Invoke-YahooChart -Symbol '^VIX' -TargetDate $targetDate -Interval $interval

  $tsT  = Convert-ToEtDatetime -UnixSeconds $respT.chart.result[0].timestamp
  $clT  = @($respT.chart.result[0].indicators.quote[0].close)
  $tsV  = Convert-ToEtDatetime -UnixSeconds $respV.chart.result[0].timestamp
  $clV  = @($respV.chart.result[0].indicators.quote[0].close)

  $rowsT = Build-SessionRows -TsEt $tsT -Close $clT -TargetDate $targetDate
  $rowsV = Build-SessionRows -TsEt $tsV -Close $clV -TargetDate $targetDate

  Write-CsvPair -TickerRows $rowsT -VixRows $rowsV -Label $label -TargetDate $targetDate -OutDir $outDir
}

Write-Host 'Done.'


