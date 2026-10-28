# 纯 skill 驱动生成准备目录

本目录准备从“原成功轨迹骨架重定向”转向“成功经验＋当前场景生成新动作”。当前只是目录、草稿和模板；未启动仿真或API调用。

## 保留的对照

旧轨迹重定向源码已保存在仓库内 [trajectory_retargeting](../trajectory_retargeting/README.md)，复制来源与哈希见其 manifest.json。原实验目录也保留，原位置和SHA-256见 [retargeting_baseline.json](retargeting_baseline.json)。现有cap-sim执行器、参考恢复、反馈、验收和成功保存逻辑均保留。

## 文件

- [pipeline-preparation.json](pipeline-preparation.json)：执行顺序、接口和输入边界。
- [skill.draft.md](skill.draft.md) / [skill.draft.json](skill.draft.json)：苹果接触经验、CaP规则与待验证策略。不是已发布skill。
- [skill_provenance.json](skill_provenance.json)：现有v0013经验的来源与边界。
- [task.template.json](task.template.json)：无references，固定mode1及失败反思、10次尝试/30秒；资产路径必须填写。
- [experiment-record.template.json](experiment-record.template.json)：正式实验前填写并冻结，不可把模板作为证据。
- [existing_code_hashes.json](existing_code_hashes.json)：准备前的现有源码哈希，供验证保留情况。

## 下一步执行顺序

1. 复核成功来源的接触证据，检查草稿只包含可迁移信息；在本目录独立registry发布新skill，不覆盖原task_skills版本。
2. 明确苹果原场景/变体、机器人和实际控制接口；补齐任务资产路径与场景几何、障碍、接触和手部视角观测。
3. 建立执行请求适配器：移除原actions/token/keyframe输入，禁止native_prefix和body_keyframes，并审计所有生成动作来源。准备文件没有实现该运行时隔离，不能现在声称已经阻断专家数据。
4. 配置明确的GPT API模型，执行环境/目标可解性/控制模式检查，填写并冻结正式实验记录；运行目录用experiments/YYYYMMDD_HHMM_skill_only_apple。
5. 先验证原场景的新生成抓取，再改变场景。GPT按skill分析当前场景、重新生成短段waypoints；控制器负责逐帧执行，反馈后纠偏。失败分析先保存，再重置重试。
6. 独立物理验收，保留成功和全部失败。当前优化预算为0；后续开启优化需单独冻结，先保存成功再优化。

总结者可见示范，执行者使用独立上下文且无原轨迹。IK可在未来独立对照中执行新waypoint，但不因此允许重用专家关节序列。外层pipeline不随试验随意修改。

本轮只完成准备；API配置、观测适配、运行时输入隔离和物理实验仍待实施。未提交Git。
