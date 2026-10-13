import json
from pathlib import Path
import statistics

root=Path('/opt/dlami/nvme/kzg-gpu-passes-20261013')
names=['warmup','baseline-1','candidate-1','candidate-2','baseline-2']
reports={name:json.loads((root/name/'pipeline/report.json').read_text()) for name in names}
reference=reports['warmup']
summary=json.loads((root/'comparison.json').read_text())
summary['trial_details']={}
summary['phase_comparison']={}
summary['proofs']=reference['proofs']
summary['controlled_variables']={
 'binaries': reference['binaries'],
 'root_artifacts': reference['root_artifacts'],
 'fri_input': reference['fri_input'],
 'host_memory_limit_gib':230,
 'cpu_affinity':reference['cpu_affinity'],
 'gpu':reference['hardware'],
 'ordering':'warmup, baseline, candidate, candidate, baseline',
 'fixed_cache_state':'warm for all four timed trials; cold worker processes',
}
for name,r in reports.items():
 assert r['status'].startswith('verified_')
 for key in ('binaries','sources','root_artifacts','fri_input','proofs'):
  assert r[key]==reference[key],(name,key)
 for key in ('first_stage_verification','recursive_verification','first_stage_cpu_verification','recursive_cpu_verification'):
  assert r[key]['native_verification_passed'] and r[key]['negative_tests_pass'],(name,key)
 scope=json.loads((root/name/'result.json').read_text())
 assert scope['returncode']==0 and not scope['timed_out']
 events=dict(line.split() for line in scope['memory_events'].splitlines())
 assert events['oom']=='0' and events['oom_kill']=='0'
 phases=r['phases']
 peaks={phase: record.get('peak_rss_bytes',record.get('memory_after',{}).get('process_lifetime_peak_rss_bytes',0))/2**30 for phase,record in phases.items()}
 recursive=phases['recursive_stage_and_prove']
 if name!='warmup':
  for phase in ('fri_stage_and_prove','recursive_stage_and_prove'):
   assert phases[phase]['fixed_cache_hits']==1 and phases[phase]['fixed_cache_misses']==0,(name,phase)
   assert phases[phase]['srs_cache_hits']>=1 and phases[phase]['srs_cache_misses']==0,(name,phase)
 summary['trial_details'][name]={
  'pipeline_seconds':r['pipeline_wall_seconds'],
  'phase_seconds':{phase:record['wall_seconds'] for phase,record in phases.items()},
  'phase_peak_rss_gib':peaks,
  'maximum_process_rss_gib':max(peaks.values()),
  'recursive_gpu_peak_gib':recursive['gpu_samples_every_2_seconds']['0']['peak_memory_mib']/1024,
  'recursive_gpu_operations':recursive['cuda_operation_totals_overlap'],
  'recursive_host_operations':recursive['host_operation_totals_overlap'],
  'scope':scope,
 }
for phase in ('fri_compress','fri_stage_and_prove','recursive_stage_and_prove'):
 baseline=[reports[f'baseline-{i}']['phases'][phase]['wall_seconds'] for i in (1,2)]
 candidate=[reports[f'candidate-{i}']['phases'][phase]['wall_seconds'] for i in (1,2)]
 a,b=statistics.median(baseline),statistics.median(candidate)
 summary['phase_comparison'][phase]={
  'baseline_samples_seconds':baseline,'candidate_samples_seconds':candidate,
  'baseline_median_seconds':a,'candidate_median_seconds':b,
  'saved_seconds':a-b,'reduction_percent':100*(a-b)/a,
 }
summary['identical_sources_binaries_inputs_and_proof_bytes']=True
summary['all_independent_cpu_verifiers_passed']=True
summary['all_timed_trials_hit_both_fixed_caches']=True
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('trial_details','controlled_variables','proofs')},indent=2))
