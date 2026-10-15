param($json, $outdir, $rate)
Add-Type -AssemblyName System.Speech
$texts = Get-Content -Raw -Encoding UTF8 $json | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Heami Desktop")
$s.Rate = [int]$rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$md5 = [System.Security.Cryptography.MD5]::Create()
$times = @()
foreach ($t in $texts) {
  $h = ($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($t)) | % { $_.ToString("x2") }) -join ""
  $p = Join-Path $outdir ($h.Substring(0,16) + ".wav")
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $s.SetOutputToWaveFile($p, $fmt); $s.Speak($t); $s.SetOutputToNull()
  $sw.Stop(); $times += $sw.Elapsed.TotalMilliseconds
}
$m = ($times | Measure-Object -Average -Maximum -Minimum)
"n=$($texts.Count) avg_ms=$([math]::Round($m.Average,1)) min_ms=$([math]::Round($m.Minimum,1)) max_ms=$([math]::Round($m.Maximum,1))"
