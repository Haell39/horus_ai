# diagnose_srt.ps1
$host = "168.90.225.116"
$port = 6052
$encodedPass = "yKz585%40353"  # @ -> %40
$srtUrlCaller = "srt://$host:$port?mode=caller&latency=4000&transtype=live&passphrase=$encodedPass&pbkeylen=16"
$srtUrlListener = "srt://:$port?mode=listener&latency=4000&transtype=live&passphrase=$encodedPass&pbkeylen=16"

"Running Test-NetConnection..." | Out-File test_netconn.txt
Test-NetConnection -ComputerName $host -Port $port | Out-File -Append test_netconn.txt

"Getting HLS playlist (if backend running)..." | Out-File hls_playlist.txt
try {
    Invoke-WebRequest http://127.0.0.1:8000/static/hls/stream.m3u8 -UseBasicParsing | Select-Object -ExpandProperty Content | Out-File -Append hls_playlist.txt
} catch {
    "Failed to fetch HLS playlist: $_" | Out-File -Append hls_playlist.txt
}

"Running ffmpeg (caller) - logs -> srt_caller.log" | Out-File srt_caller.log
# 10s recording caller mode
& ffmpeg -loglevel debug -fflags +genpts -rw_timeout 10000000 -i $srtUrlCaller -t 10 -c copy .\srt_test_caller.mp4 2>> srt_caller.log

"Running ffmpeg (listener) - logs -> srt_listener.log" | Out-File srt_listener.log
# 10s recording listener mode (requires remote to connect)
& ffmpeg -loglevel debug -fflags +genpts -i $srtUrlListener -t 10 -c copy .\srt_test_listener.mp4 2>> srt_listener.log

"Done. Collected files: test_netconn.txt, hls_playlist.txt, srt_caller.log, srt_listener.log" | Out-File results_summary.txt