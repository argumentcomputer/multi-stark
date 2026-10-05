import os, pathlib, resource, subprocess, time
root=pathlib.Path('/home/arthur/multi-stark')
run=root/'target/init-kzg-run'
print('Waiting for full Init staging to finish',flush=True)
while True:
    status=(run/'stage.time').read_text()
    if 'Exit status:' in status:
        if 'Exit status: 0' not in status or not (run/'data/manifest.bin').exists():
            raise SystemExit('Staging failed; proof not started')
        break
    time.sleep(5)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_AS,(440<<30,440<<30))
print('Starting full Init KZG prover',flush=True)
with (run/'prove.log').open('w') as log:
    result=subprocess.run(['/usr/bin/time','-v','-o',str(run/'prove.time'),str(root/'target/release/prove'),str(run/'data')],cwd=root,env=dict(os.environ,RAYON_NUM_THREADS='32'),stdout=log,stderr=subprocess.STDOUT)
print('Prover exit status:',result.returncode,flush=True)
raise SystemExit(result.returncode)
