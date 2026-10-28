"""Generate an honest local report; never manufactures missing cases or evidence."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'audit_grasps.py')],check=True)
subprocess.run([sys.executable,str(r/'audit_results.py')],check=True)
a=json.loads((r/'results_audit.json').read_text());rows=a['cases']
lines=['# HUMANO 10-task CaP experiment','',f'Updated: {time.strftime("%Y-%m-%d %H:%M:%S %z")}', '', 'Repository baseline: `2ed4500`; native integration: `d991dd5`.', '', 'This is a current-session GPT / GPT-authored CaP experiment, not a live external-model API benchmark. No-reference cases precede expert rollout access. Each group has 10 tasks; each case allows at most 10 attempts, 30 simulated seconds per attempt, 12 phase feedback calls and one post-success optimization within the same budget.', '', 'Reference variant: object +0.10 m along world Y. New lateral alignment and reference wrist retargeting use the same SONIC control interface. Native criteria and physics are fixed.','']
for group in ['no_reference','reference_augmented']:
 xs=[x for x in rows if x['group']==group]
 lines += [f'## {group}', '', f'Completed: {len(xs)}/10. Valid task successes: {sum(x["valid_task_success"] for x in xs)}/10. Native passes: {sum(x["native_success"] for x in xs)}/10. Missing cases are pending, not failures.','', '| Task | Valid / native pass | Attempts | Selected attempt / sim s | Interrupted slots | Videos |','|---|---|---:|---|---|---|']
 for x in xs:
  v=r/'videos'/group/x['task'];links=' / '.join(f'[{k}]({(v/(k+".mp4")).relative_to(r)})' for k in ['ego','external']) if x['video_verified'] else 'pending'
  lines.append(f'| {x["task"].removeprefix("G1Wholebody").removesuffix("Teleop")} | {x["valid_task_success"]} / {x["native_success"]} | {x["attempts"]} | {x["selected_attempt"]} / {round(x["selected_sim_s"],3) if x["selected_sim_s"] else "—"} | {x["interrupted_attempts"]} | {links} |')
 lines+=['']
lines+=['## Reference provenance', '', '| Task | Episode | Saved-state native check | First pass s |', '|---|---:|---|---:|']
for p in sorted((r/'references').glob('G1*.json')):
 ref=json.loads(p.read_text());lines.append(f'| {ref["task"]} | {ref["episode"]} | {ref["native_success_from_saved_states"]} | {ref["first_success_s"]} |')
lines+=['', 'Source sim.ctrl is absent. Reference contact forces were estimated by PD reconstruction from recorded action.wbc and sampled qpos/qvel, not exact measured torque/contact records. A saved-state pass is not a demonstrated successful controller replay. ChairSit, TableApplePick and TableButtonPress remained unverified after two candidate references; their runs are reference-conditioned generation, not verified-success-experience transfer.', '']
verified_names={p.stem for p in (r/'references').glob('G1*.json') if json.loads(p.read_text())['native_success_from_saved_states']}
ref_rows=[x for x in rows if x['group']=='reference_augmented']
lines += [f'Reference cases with saved-state native pass: {sum(x["valid_task_success"] for x in ref_rows if x["task"] in verified_names)} valid generated successes / {len(verified_names)} source cases. Unverified-reference cases: {sum(x["valid_task_success"] for x in ref_rows if x["task"] not in verified_names)} / {10-len(verified_names)}. Pending executions are not completed failures.', '', 'Policy revisions are recorded in policy_revision.json, policy_revision_2.json and policy_revision_3.json. Target-specific bilateral contact gating was corrected before any reference pick attempt. Later task retries may switch from full-body reference targets to the existing Cartesian-upper-body plus SONIC-leg-planner API after measured tracking failure; post-reference lifts may add 3.5cm only while actual target contacts persist. No physical/controller/native acceptance parameters were changed.', '']
download=json.loads((r/'download_summary.json').read_text())
lines += [f'Calendar elapsed since download start: {time.time()-download["started_unix_s"]:.3f} s at report generation. This includes setup, idle/session gaps and interrupted work; it is not active simulation time.', '']
lines+=['## Timing and limits','','Download wall time: 154.300 s. Attempt wall times include policy decisions, manual mailbox waits, simulation and evidence export/verification; they are not pure physics times. Decision/mailbox records have different coverage, including interrupted work, so that column is neither a subset of nor additive to completed-attempt wall time. The machine/session had infrastructure interruptions and idle gaps. Summed task work is not calendar elapsed time when runs overlap. Per-case `results_audit.json`, mailbox timing files, `direct_execution.json`, reference extraction records and per-video `render_timing.json` preserve the measured components.','','| Group | Recorded completed-attempt wall s | Decision/mailbox wall s | Executed sim s |','|---|---:|---:|---:|']
for group in ['no_reference','reference_augmented']:
 xs=[x for x in rows if x['group']==group];lines.append('| '+group+' | '+' | '.join(f'{sum(x[k] for x in xs):.3f}' for k in ['recorded_attempt_wall_s','mailbox_decision_wall_s','sim_s_executed'])+' |')
lines+=['','Native pick checks detect height only. The basket lifted by knee collision is excluded from valid grasp/task successes; raw native success and its trajectory remain preserved. Process-loss/timeouts are reported separately and are not evidence of controller task failure.','','Videos replay saved measured states through MuJoCo EGL, using head stereo left (first person) and a fixed external (third person) camera. MP4 full-decode and sampled-frame checks are recorded. They are not Isaac renders or certified training-ready video.','','Failures under this budget mean this implementation did not pass the case; they do not establish that KPI or skill representations are ineffective. Native reference saved-state acceptance is not a guarantee that the reference will replay successfully through this controller.','','## Failure records','']
for x in rows:
 lines.append(f'- {x["group"]}/{x["task"]}: '+('; '.join(f"attempt {f['attempt']}: {f['reason']}" for f in x['failures']) or 'none'))
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(r/'REPORT.md')
