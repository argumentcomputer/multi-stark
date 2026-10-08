"""Run checkpointed FRI-to-KZG compression with a process memory limit."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 2 and sys.argv[2] == '--collect-when-done':
    while not (out / 'verify.complete.json').exists():
        for result_path in out.glob('*.json'):
            result = json.loads(result_path.read_text())
            if result.get('returncode', 0):
                raise SystemExit(f'Pipeline failed: {result_path}')
        time.sleep(10)
binary = out / 'init_fri_kzg_prove'
if not binary.exists():
    shutil.copy2('target/release/examples/init_fri_kzg_prove', binary)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

source = ['examples/init_fri_kzg_prove.rs', 'examples/support/init_fri.rs', 'src/ark_adapter/config.rs', 'src/ark_adapter/pcs.rs', 'src/prover.rs', 'src/system.rs']
inputs = [Path('target/init-fri-wrap') / n for n in ['outer-proof.bin', 'outer-vk.bin', 'outer-claims.bin']]
provenance = {'binary_sha256': digest(binary), 'inputs': {str(p): digest(p) for p in inputs}, 'source': {n: digest(Path(n)) for n in source}, 'development_srs': True}
if (out / 'provenance.json').exists():
    recorded = json.loads((out / 'provenance.json').read_text())
    assert recorded['binary_sha256'] == provenance['binary_sha256'] and recorded['inputs'] == provenance['inputs'], 'binary or input changed; use a new output directory'
else:
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

def limits():
    cap = 450 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

for phase, args in [('stage', ['stage', 'target/init-fri-wrap', str(out / 'data')]), ('prove', ['prove', str(out / 'data')]), ('verify', ['verify', str(out / 'data')])]:
    marker = out / (phase + '.complete.json')
    if marker.exists():
        continue
    if phase == 'stage' and (out / 'data/manifest.bin').exists():
        continue
    stamp = str(time.time_ns())
    started = time.time()
    with (out / (phase + '-' + stamp + '.log')).open('w') as log, (out / (phase + '-' + stamp + '.time')).open('w') as timing:
        process = subprocess.Popen(['/usr/bin/time', '-v', str(binary), *args], stdout=log, stderr=timing, env=dict(os.environ, RAYON_NUM_THREADS='32'), preexec_fn=limits)
        print(f'{phase} started, log {log.name}', flush=True)
        code = process.wait()
    result = {'phase': phase, 'returncode': code, 'wall_seconds': time.time() - started, 'log': log.name, 'timing': timing.name}
    (out / (phase + '-' + stamp + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    if code:
        raise SystemExit(code)
    marker.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{phase} completed in {result["wall_seconds"]:.1f}s', flush=True)

import re
phases = {}
for phase in ['stage', 'prove', 'verify']:
    result = json.loads((out / (phase + '.complete.json')).read_text())
    timing = Path(result['timing']).read_text()
    result['peak_rss_bytes'] = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', timing)[1]) * 1024
    phases[phase] = result
files = {name: {'bytes': (out / 'data/kzg' / name).stat().st_size, 'sha256': digest(out / 'data/kzg' / name)} for name in ['proof.compact.bin', 'packet.bin', 'profile-id.bin']}
report = {
    'status': 'generated_and_independently_verified',
    'production_setup': False,
    'security': 'Known-trapdoor development SRS; correctness and cost experiment only.',
    'layout': 'One ordinary KZG proof over height-capped traces, including compact BLAKE3 and merged tables.',
    'public_claim': 'Original 18 Init root public words; all 18 altered words rejected.',
    'phases': phases,
    'artifacts': files,
    'verification': (out / 'data/kzg/VERIFIED.txt').read_text().strip(),
    'timing_notes': 'prove includes SRS generation, circuit setup, checkpoint writes, proving and verification. verify is a separate process including SRS generation; the verification line times verification of an already decoded proof.',
    'provenance': json.loads((out / 'provenance.json').read_text()),
}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'artifacts': files}, indent=2), flush=True)
