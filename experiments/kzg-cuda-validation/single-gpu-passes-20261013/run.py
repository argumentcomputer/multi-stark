#!/usr/bin/env python3
"""Compare bounded GPU consumers with the host fallback using identical proof inputs."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

REPO = Path('/home/sam/repos/multi-stark')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def child(root, name, mode, timeout):
    output = root / name
    output.mkdir()
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('MULTI_STARK_', 'RAYON_', 'OMP_')) or key == 'CUDA_VISIBLE_DEVICES':
            del env[key]
    env.update({
        'CUDA_VISIBLE_DEVICES': '0', 'MULTI_STARK_CUDA_DEVICE': '0',
        'MULTI_STARK_CUDA_AUX_DEVICES': '', 'MULTI_STARK_CUDA_MEMORY_LOG': '1',
        'MULTI_STARK_KZG_CUDA_SRS_CACHE': '1', 'MULTI_STARK_KZG_CUDA_PASSES': mode,
        'MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP': '0',
        'MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT': '0',
    })
    command = [
        sys.executable, str(REPO / 'experiments/kzg-cuda-bench.py'),
        '--output', str(output / 'pipeline'),
        '--root-artifacts', str(REPO / 'experiments/kzg-cuda-validation/resume-20261009/root-artifacts'),
        '--setup', 'development', '--development-public-degree', '--sequential-workers',
        '--kzg-devices', '0', '--fixed-cache', str(root / 'fixed-cache'),
        '--srs-cache', '/opt/dlami/nvme/kzg-single-gpu-pilot-20261013/srs-cache',
        '--first-stage-binary', str(root / 'bin/init_fri_kzg_prove'),
        '--wrapper-binary', str(root / 'bin/init-kzg-wrap'),
        '--fri-binary', str(root / 'bin/ix_root'),
    ]
    if name != 'warmup':
        command += ['--baseline', str(root / 'warmup/pipeline')]
    cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    def read(name):
        try:
            return (cgroup / name).read_text().strip()
        except OSError:
            return None
    save(output / 'metadata.json', {
        'command': command, 'passes': mode, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cgroup': str(cgroup), 'memory_limit_bytes': read('memory.max'),
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'timeout_seconds': timeout,
    })
    started = time.monotonic()
    timed_out = False
    with (output / 'runner.log').open('w') as log, (output / 'memory.jsonl').open('w', buffering=1) as samples:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            stat = dict(line.split() for line in (read('memory.stat') or '').splitlines())
            samples.write(json.dumps({
                'elapsed_seconds': time.monotonic() - started, 'memory_bytes': read('memory.current'),
                'anon_bytes': stat.get('anon'), 'file_bytes': stat.get('file'),
                'memory_events': read('memory.events'),
            }) + '\n')
            if time.monotonic() - started > timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        result = {
            'returncode': process.wait(), 'timed_out': timed_out,
            'wall_seconds': time.monotonic() - started, 'memory_peak_bytes': read('memory.peak'),
            'memory_events': read('memory.events'), 'memory_pressure': read('memory.pressure'),
        }
    save(output / 'result.json', result)
    if result['returncode'] or timed_out:
        raise RuntimeError(f'{name} failed: {result}')
    report = json.loads((output / 'pipeline/report.json').read_text())
    recursive = report['phases']['recursive_stage_and_prove']
    operations = recursive['cuda_operation_totals_overlap']
    observed = {op: operations.get(op, {}).get('calls', 0) for op in ('lookup-passes', 'quotient-passes')}
    if (mode == '1' and any(count != 1 for count in observed.values())) or (mode == '0' and any(observed.values())):
        raise RuntimeError(f'{name}: unexpected GPU dispatch: {observed}')
    print(f'{name}: verified {report["pipeline_wall_seconds"]:.3f}s; dispatch={observed}', flush=True)


def compare(root, budget):
    root.mkdir(exist_ok=False)
    (root / 'bin').mkdir()
    binaries = {
        'init_fri_kzg_prove': REPO / 'target/release/examples/init_fri_kzg_prove',
        'ix_root': REPO / 'target/release/examples/ix_root',
        'init-kzg-wrap': REPO / 'experiments/kzg-wrap/target/release/init-kzg-wrap',
    }
    for name, path in binaries.items():
        shutil.copy2(path, root / 'bin' / name)
    shutil.copy2(__file__, root / 'run.py')
    save(root / 'binary-sha256.json', {name: hashlib.file_digest((root / 'bin' / name).open('rb'), 'sha256').hexdigest() for name in binaries})
    started = time.monotonic()
    order = [('warmup', '1'), ('baseline-1', '0'), ('candidate-1', '1'), ('candidate-2', '1'), ('baseline-2', '0')]
    for name, mode in order:
        remaining = budget - (time.monotonic() - started)
        if remaining < 300:
            raise RuntimeError('Comparison budget exhausted before the next complete-proof trial')
        print(f'Starting {name}, passes={mode}; remaining budget {remaining:.0f}s', flush=True)
        subprocess.run([
            'systemd-run', '--user', '--scope', '--quiet', '--collect',
            '--unit', f'kzg-passes-{os.getpid()}-{name}',
            '-p', 'MemoryMax=230G', '-p', 'MemorySwapMax=0', '-p', 'OOMPolicy=continue',
            sys.executable, str(root / 'run.py'), '--output', str(root), '--child', name,
            '--passes', mode, '--budget-seconds', str(int(remaining)),
        ], check=True)
    trials = {name: json.loads((root / name / 'pipeline/report.json').read_text()) for name, _ in order}
    baseline = [trials[f'baseline-{i}']['pipeline_wall_seconds'] for i in (1, 2)]
    candidate = [trials[f'candidate-{i}']['pipeline_wall_seconds'] for i in (1, 2)]
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    result = {
        'order': order, 'baseline_seconds': baseline, 'candidate_seconds': candidate,
        'baseline_median_seconds': baseline_median, 'candidate_median_seconds': candidate_median,
        'saved_seconds_per_proof': baseline_median - candidate_median,
        'reduction_percent': 100 * (baseline_median - candidate_median) / baseline_median,
        'total_comparison_seconds': time.monotonic() - started,
        'scope': 'one warmup excluded, two paired full root-to-packet runs per variant, cached SRS and fixed data, cold worker processes',
        'profile': 'development public-degree v4; not Filecoin acceptance',
    }
    save(root / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child')
    parser.add_argument('--passes', choices=('0', '1'))
    parser.add_argument('--budget-seconds', type=int, default=2700)
    args = parser.parse_args()
    if args.child:
        child(args.output, args.child, args.passes, args.budget_seconds)
    else:
        compare(args.output, args.budget_seconds)
