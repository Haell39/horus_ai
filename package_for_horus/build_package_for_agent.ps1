# PowerShell script to collect model artifacts into this package directory
$base = Split-Path -Parent $MyInvocation.MyCommand.Definition

# targets to copy (source -> dest subfolder)
$items = @(
    @{ src = "models/audio_mobilenetv2/model_retrain_dryrun.h5"; dst = "audio" },
    @{ src = "models/audio_mobilenetv2/model_retrain_dryrun.keras"; dst = "audio" },
    @{ src = "models/audio_mobilenetv2/checkpoints"; dst = "audio/checkpoints" },
    @{ src = "models/audio_mobilenetv2/metadata.json"; dst = "audio" },
    @{ src = "models/audio_mobilenetv2/thresholds.yaml"; dst = "audio" },
    @{ src = "outputs/results/training_summary_phase2_audio.txt"; dst = "audio" },
    @{ src = "outputs/results/training_history_audio.json"; dst = "audio" },

    @{ src = "models/vision_mobilenetv2/model_retrain_dryrun.h5"; dst = "vision" },
    @{ src = "models/vision_mobilenetv2/model_retrain_dryrun.keras"; dst = "vision" },
    @{ src = "models/vision_mobilenetv2/checkpoints"; dst = "vision/checkpoints" },
    @{ src = "models/vision_mobilenetv2/metadata.json"; dst = "vision" },
    @{ src = "models/vision_mobilenetv2/thresholds.yaml"; dst = "vision" },
    @{ src = "outputs/results/training_summary_phase2_vision.txt"; dst = "vision" },
    @{ src = "outputs/results/training_history_vision.json"; dst = "vision" }
)

foreach ($it in $items) {
    $src = Join-Path -Path (Get-Location) -ChildPath $it.src
    $dstDir = Join-Path -Path $base -ChildPath $it.dst
    if (Test-Path $src) {
        Write-Host "Copying $src -> $dstDir"
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        if ((Get-Item $src).PSIsContainer) {
            Copy-Item -Path $src -Destination $dstDir -Recurse -Force
        } else {
            Copy-Item -Path $src -Destination (Join-Path $dstDir (Split-Path $src -Leaf)) -Force
        }
    } else {
        Write-Host "MISSING: $src (skipping)"
    }
}

Write-Host "Package build complete. Files are under $base\audio and $base\vision"