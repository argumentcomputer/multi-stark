import os,pathlib,resource,subprocess,sys,time,json,hashlib
root=pathlib.Path('/home/arthur/multi-stark')
dir=pathlib.Path(sys.argv[1]).resolve(); logs=dir/'kzg'; logs.mkdir(exist_ok=True)
def report(phase,status="running",error=None):
    if dir != root/'target/init-kzg-run/data': return
    path=root/'target/init-kzg-run/run.json'
    record=json.loads(path.read_text())
    record.update(status=status,phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        completed_setup=len(list(logs.glob('setup-*.bin'))),completed_headers=len(list(logs.glob('header-*.bin'))),completed_proofs=len(list(logs.glob('proof-*.bin'))))
    if error: record['error']=error
    if status=='full_root_verified_development_srs':
        record['proof_bytes']=(logs/'proof.bin').stat().st_size
        record['proof_file']=str(logs/'proof.bin')
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)

def start(mode,worker,workers,limit):
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_AS,(limit<<30,limit<<30))
    log=(logs/f'{mode}-{worker}.log').open('w')
    env=dict(os.environ,RAYON_NUM_THREADS='32',KZG_RUN_MODE=mode,KZG_RUN_WORKER=str(worker),KZG_RUN_WORKERS=str(workers))
    p=subprocess.Popen(['/usr/bin/time','-v','-o',str(logs/f'{mode}-{worker}.time'),str(root/'target/release/prove'),str(dir)],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
    log.close();return p
digest=hashlib.sha256(b'init-root-full-kzg-sharded-v1;multi-stark/kzg/v3;quotient=8')
files=[dir/'manifest.bin',*sorted(dir.glob('*.meta')),*sorted(dir.glob('*.zst'))]
for path in files:
    digest.update(path.name.encode()+b'\0')
    digest.update(path.stat().st_size.to_bytes(8,'little'))
    with path.open('rb') as f:
        while chunk:=f.read(1<<20): digest.update(chunk)
job_id=digest.hexdigest()
seal=logs/'job-sha256.txt'
if seal.exists() and seal.read_text().strip()!=job_id:
    raise SystemExit('Staged input differs from the checkpointed job')
seal.write_text(job_id+'\n')
print('Checkpoint input digest:',job_id,flush=True)
for mode in ['setup','headers','prove','verify']:
    report(mode)
    n=1 if mode=='verify' else 2
    jobs=[start(mode,i,n,220) for i in range(n)]
    last=['']*n
    while any(p.poll() is None for p in jobs):
        for i,p in enumerate(jobs):
            lines=(logs/f'{mode}-{i}.log').read_text().splitlines()
            progress=[l for l in lines if not l.startswith('VmSize:')]
            if progress and progress[-1]!=last[i]:
                last[i]=progress[-1];print(f'{mode} worker {i}: {last[i]}',flush=True)
        report(mode)
        time.sleep(5)
    for i,p in enumerate(jobs):
        if p.returncode:
            text=(logs/f'{mode}-{i}.log').read_text()
            if 'memory allocation' not in text:
                report(mode,'failed',f'{mode} worker {i} failed')
                raise SystemExit(f'{mode} worker {i} failed: see {logs/f"{mode}-{i}.log"}')
            print(f'{mode} worker {i}: retrying alone with 440 GiB limit',flush=True)
            # Preserve the failed attempt's diagnostics.
            (logs/f'{mode}-{i}.log').rename(logs/f'{mode}-{i}.memory-failure.log')
            retry=start(mode,i,n,440)
            if retry.wait():raise SystemExit(f'{mode} worker {i} retry failed')
    print(f'{mode}: complete',flush=True)
report('complete','full_root_verified_development_srs')
print((logs/'VERIFIED.txt').read_text(),flush=True)
