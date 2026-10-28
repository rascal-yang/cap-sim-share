#!/usr/bin/env python3
"""Read-only numerical analysis and standalone report for paired rollback trials."""
import argparse,json,statistics
from pathlib import Path


def analyze(root):
    root=Path(root);s=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text());budget=config['benchmark']['wall_budget_s'];rows=[];pairs=[]
    for task in s['results']:
        shared=task.get('shared_cost_s',0)
        for co in task['conditions']:
            pair={'profile':task['profile'],'repeat':co['repeat']}
            for arm,a in co['arms'].items():
                eligible=[r for r in a['rows'] if r['elapsed_s']+shared<=budget];new=[r for r in eligible if not r.get('shared_seed')]
                raw=[r for r in a['rows'] if r['elapsed_s']+task['planning_wall_s']+task['seed']['attempt_wall_s']<=budget]
                out={'profile':task['profile'],'repeat':co['repeat'],'arm':arm,'budget_s':budget,'shared_cost_s':shared,'task_success':any(r['success'] for r in eligible),'first_success_s':min((r['elapsed_s']+shared for r in eligible if r['success']),default=None),'new_attempts':len(new),'new_successes':sum(r['success'] for r in new),'new_attempt_success_rate':sum(r['success'] for r in new)/len(new) if new else None,'full_planning_charge_success':any(r['success'] for r in raw),'full_planning_charge_new_attempts':sum(not r.get('shared_seed') for r in raw),'overrun_attempts':len(a['rows'])-len(eligible),'actual_arm_wall_s':a['arm_wall_s'],'queue_exhausted':a['queue_exhausted'],'mean_new_physics_wall_s':statistics.mean(r['result']['simulation_wall_s'] for r in new) if new else None,'mean_new_total_wall_s':statistics.mean(r['attempt_wall_s'] for r in new) if new else None,'curves':a['curves']};rows.append(out);pair[arm]=out
            if len(co['arms'])==2:pairs.append(pair)
    summary=[]
    for profile in [t['profile'] for t in s['results']]:
        group={arm:[r for r in rows if r['profile']==profile and r['arm']==arm] for arm in ('restart','rollback')}
        summary.append({'profile':profile,'arms':{arm:{'conditions':len(rs),'successful_conditions':sum(r['task_success'] for r in rs),'new_attempts':sum(r['new_attempts'] for r in rs),'new_successes':sum(r['new_successes'] for r in rs),'mean_first_success_s':statistics.mean(v) if (v:=[r['first_success_s'] for r in rs if r['first_success_s'] is not None]) else None,'full_charge_successful_conditions':sum(r['full_planning_charge_success'] for r in rs)} for arm,rs in group.items()}})
    analysis={'budget_s':budget,'conditions':rows,'pairs':pairs,'summary':summary,'scope':'three tasks, one fixed scene each, three candidate-order permutations; paired deterministic search, not independent scenes','success_definition':'native/approved-profile success; only selected winner per task receives full/token proof','shared_seed_not_new_attempt':True};(root/'analysis.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False))
    lines=[f"# {root.name}",f"\n实际运行：{s['started_at']} → {s['ended_at']}。数值阶段结束：{s['numeric_ended_at']}。",'\n目标：检验完整控制器状态回溯是否在相同时间预算内提高成功率、减少时间并增加尝试次数。', '\n设计：坐椅子、按按钮、拿苹果，各1个固定历史场景（物体X+5mm），每任务3种预先固定的候选顺序。真实gpt-6-astra low为每任务生成1组8个参数候选；两组共享队列和初始种子。原始token前缀仍保留，属于轨迹重定向。每组120s，双方计入相同的规划摊销成本（实际规划÷3）与完整种子生成/验收成本。独立规划全额收费的结果同时报告。物理准备、保存、候选验收计入搜索；渲染、实现等价检查、最终完整重启/token证明单列。', '\n达到成功后继续同一队列以衡量批量产出；到期只停止启动新尝试，已启动尝试保留证据，但结束超时不算预算内尝试/成功。共享种子不是新物理尝试，不重复计入新尝试/新成功数量。3种顺序不是3个独立场景。', '\n|任务|方式|120秒内成功条件数|新尝试数|新验收成功数|首次成功平均s（成功条件）|全额规划成功条件数|\n|---|---|---:|---:|---:|---:|---:|']
    for task in summary:
        for arm,d in task['arms'].items():lines.append(f"|{task['profile']}|{arm}|{d['successful_conditions']}/{d['conditions']}|{d['new_attempts']}|{d['new_successes']}|{d['mean_first_success_s'] if d['mean_first_success_s'] is not None else '未成功'}|{d['full_charge_successful_conditions']}/{d['conditions']}|")
    lines+=['\n“新验收成功”是原生或已批准按钮验收的暂定成功，并非每条都独立验证过token回放。仅每任务选取首个预算内成功回溯候选做完整重启＋独立token/双手回放。按钮使用既有button_palm_mesh_v1：添加精确手掌碰撞网格，仍要求20N、1mm、50帧；不把腕壳计入，原生验收另存。', '\n改动：保存/恢复腿部反馈积分、手腕力反馈积分、手指积分和上一帧命令；保留原物理积分状态、solver warmstart、SONIC历史、接触验收状态。完整拼接继承双手命令，checkpoint manifest绑定手部及反馈日志。参数变化只能回到最早受影响时刻之前；相同参数可以恢复接触中的检查点。固定计划由本地编译器从API参数补丁组装，来源kind=api_compiled，不伪称完整计划是API原文。', '\n执行验证：']
    for task in s['results']:
        eq=task.get('equivalence',[]);v=task.get('verification');lines.append(f"\n- {task['profile']}: status={task['status']}；早期/接触中检查点完全一致={[(e['frame'],e['proof']['exact']) for e in eq]}；选定成功完整重启/token证明={v['passed'] if v else '未产生可验证胜者'}。")
        if task.get('error'):lines.append('  错误：'+task['error'])
    lines += [f"\nAPI请求{s['api_requests']}次；已知服务端token={s['known_tokens']}，未知用量请求={s['unknown_usage_calls']}。数值阶段{s['numeric_wall_s']:.3f}s，总墙钟{s['total_wall_s']:.3f}s；渲染{sum(v.get('rendering_wall_s',0) for v in s['videos']):.3f}s。开发/手工规划耗时未独立计量，未知。此前失败预检查若有，见config.recovery_of，成本不冒充本轮搜索成本。", '\n运行环境：scripts/humano_python.sh，TRT10.13.0.35、ORT1.23.2、SONIC1.0 CPU、MuJoCo3.3.6。每次执行前冻结完整experiment.json和实际provider/model/code/skill证据，原始token前缀与实际mode记录在controller.npz。所有身体PD target保持decoder输出，手指独立命令保存。新代码在独立augmentation模块中，历史其他路线未改。', '\n解释限制：这项实验衡量共享批量规划模式，不能直接推出独立实时API规划的端到端收益。初始化及证据写入可能抵消物理步数节省；必须看实际墙钟。失败分支不能当成未执行的零误差。成功率结果受候选顺序和具体任务制约，不能外推全部HUMANO任务；不使用这9个顺序条件宣称统计显著。', '\n视频（记录状态重渲染，用于验收，不代表新增物理成功）：']
    for v in s['videos']:
        if v.get('video'):lines.append(f"\n- [{v['profile']} {v['label']}]({v['video']})")
    lines+=['\n复现：scripts/run_rollback_benchmark.py读取本目录config.json/provider.json；新运行必须使用新实验目录并更新config_path/spec_paths，不能覆盖当前冻结证据。单独汇总：scripts/report_rollback_benchmark.py --root <目录>。完整数值见analysis.json，逐候选控制与参数见progress.json、各trial.json/compiled_plan.json。视频质量单独记录；training_ready=false。']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return analysis


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();out=analyze(a.root);print(json.dumps(out['summary'],indent=2))
