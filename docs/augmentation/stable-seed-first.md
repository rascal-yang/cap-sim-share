# 稳定版恢复：先复验已知可行解

恢复依据：20261023_1102_api_ten_tasks_rollback_validation。该版同一X+5mm条件下坐椅子、拿苹果、按按钮均首次通过；不能称为十任务全成功。

退化版本保存在Git `71f7c28`及分支 `archive/api-autonomous-before-stable-restore`。恢复的文件及快照哈希见stable_restore_manifest.json。三API轮换保留；自主查询助手文件留作历史，但默认contact编排不再使用探索循环。

唯一行为改动：在task配置中提供`verified_seed`，指向原API接受的attempt/result.json。先核对API请求/响应来源、审查绑定、原生成功、源轨迹、标注、安全条件、目标偏移及源证据哈希，再用原API决策重新执行完整物理轨迹。绝不以旧success代替新执行。

若新执行通过原生与独立验收，且场景及state/next_state/commands与旧成功轨迹逐项完全一致，允许标记`local_exact_seed_revalidation`。此路径无需新API规划或复核，视频和导出标明本地等价复验，training_ready仍为false。若失败或有差异，进入稳定版API复核/修复流程。原成功记录不可被失败候选覆盖。

当前严格限定相同条件；不同任务/episode/shift或未经原API接受的方案会拒绝作为seed。新扰动应使用正常API规划，不能把同条件复验当作泛化成功。未提供seed的任务使用恢复后的原流程。

入口仍为scripts/run_contact_batch.py，三个凭据通过--credentials、--secondary-credentials、--additional-credentials传入。默认成功方案登记见verified_seeds.json；每次新测试新建月/日/分钟实验目录并冻结契约，不修改历史实验。

验证结果：164项测试通过；[20261023_1919_stable_seed_regression](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1919_stable_seed_regression/REPORT.md)三任务各一次新物理执行均通过，场景和完整状态/控制数组与原成功案例完全一致。总耗时118.19秒，新API请求及token均为0。结论限于同条件恢复。
