$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$Output = Join-Path $Root 'data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl'
$Manifest = "$Output.manifest.json"
$Rejects = "$Output.rejects.jsonl"
$Log = Join-Path $Root 'logs/fresh_eval_low_topup_chatgpt_s46.log'
$Status = Join-Path $Root 'logs/fresh_eval_low_topup_chatgpt_s46.status'
$PidFile = Join-Path $Root 'logs/fresh_eval_low_topup_chatgpt_s46.pid'

if ((Test-Path -LiteralPath $Output) -or
    (Test-Path -LiteralPath $Manifest) -or
    (Test-Path -LiteralPath $Rejects)) {
    @(
        'REFUSED'
        'reason=output_exists'
        "ended_at=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    ) | Set-Content -LiteralPath $Status -Encoding ascii
    exit 2
}

$PID | Set-Content -LiteralPath $PidFile -Encoding ascii
@(
    'RUNNING'
    "started_at=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    'execution_host=local_windows'
    'provider=azure'
    'model=gpt-chat-latest'
    'target=200'
    'seed=46'
) | Set-Content -LiteralPath $Status -Encoding ascii

$env:PYTHONUNBUFFERED = '1'
$Arguments = @(
    '-m', 'dotenv', '-f', (Join-Path $Root 'data.env'), 'run', '--',
    'python', (Join-Path $Root 'generate_fresh_eval_tasks.py'),
    '--num_tasks', '200',
    '--oversample', '8',
    '--providers', 'azure',
    '--azure_models', 'gpt-chat-latest',
    '--benchmark', (Join-Path $Root 'data/testing/grpo_data_graphv2.jsonl'),
    '--synthetic', (Join-Path $Root 'data/datasets/synthetic_pool_graphv2.jsonl'),
    '--decontam_jsonl', (Join-Path $Root 'data/testing/fresh_eval_llm.jsonl'),
    '--decontam_jsonl', (Join-Path $Root 'data/testing/fresh_eval_lowmid_topup_s44.jsonl'),
    '--decontam_jsonl', (Join-Path $Root 'data/testing/fresh_eval_low_topup_deepseek_s45.jsonl'),
    '--out', $Output,
    '--workers', '4',
    '--jac_thr', '0.55',
    '--seq_thr', '0.70',
    '--strata_mix', 'low:1,mid:0,high:0',
    '--stability_runs', '2',
    '--mutation_max', '8',
    '--mutation_min_kill', '0.5',
    '--shape_gate', '1',
    '--loc_tol', '1.0',
    '--branch_tol', '1.0',
    '--rng_seed', '46',
    '--dart_bin', 'C:\flutter\bin\cache\dart-sdk\bin\dart.exe'
)

& python @Arguments *>> $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    @(
        'DONE'
        'exit_code=0'
        "ended_at=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
        'execution_host=local_windows'
        'provider=azure'
        'model=gpt-chat-latest'
        'target=200'
        'seed=46'
    ) | Set-Content -LiteralPath $Status -Encoding ascii
} else {
    @(
        'FAILED'
        "exit_code=$ExitCode"
        "ended_at=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
        'execution_host=local_windows'
        'provider=azure'
        'model=gpt-chat-latest'
        'target=200'
        'seed=46'
    ) | Set-Content -LiteralPath $Status -Encoding ascii
}

exit $ExitCode
