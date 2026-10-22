# TUF-E 自学习智能体设计语言 v0.1

> ⚠️ **诚实状态标注（2026-10-22 追加，v1.15 之后）：本文档是历史起点，不是当前蓝图。**
>
> 本 v0.1「生成性规范」自 2026-10-18 定稿后未再改动；此后所有新约束与实验发现都累积在
> **docs/tufe-design-language-v03.md（＝当前活蓝图）**。读 v0.1 前须知三条已被后续实测**订正 / 取代**：
>
> 1. **R7「拓扑 / Betti 变化 ＝ 学到新东西」已被取代。** 外部同行评审实测：发布引擎的「相变」是
>    **Betti churn**（拓扑噪声，非学习）。当前实际使用的学习信号是 **fire-viable 相对背景 ρ 的超额
>    因果有效性（OC-9）**——v0.1 未包含此判据。
> 2. **R3「跨观察者共识」从普适核心降级为域受限机制。** v1.4 Ising 连续域实测共识产生**负效应**
>    （A<B<C）。共识不是普适生长条件，适用域受限（见 v0.3 的 OC-1 / OC-3）。
> 3. **v0.1 缺失当前真正扛事的机制。** 最近的干净正例（v1.15，回应评审 ask #4）赢在**因果记忆寻址 +
>    骨架粒度（SC-1 / SC-2）**，并受 **OC-12 信息门**、**F1 饱和天花板 / F2 中漂移脆弱共振**约束——
>    这些 v0.1 里一条都没有，全在 v0.3。
>
> **未被推翻的地基（仍成立）：** 核心枢机数学、三个「独一份」本体主张（知识＝耗散结构 / 学习＝可行空间
> 维持 / 非奖励驱动）、D1 ≠ RL、D2 ≠ LLM 微调、D3 ≠ 符号 AI、R2 耗散平衡、R4 完整工作循环、R6 可行空间导向。
>
> **「完整重写本生成性规范」＝ 已登记的待触发里程碑**，触发条件（写死）：**F1 / F2 至少攻克一条 +
> 可持续内生结晶首次跑通**。在此之前不定稿——给一台尚未造好的机器画竣工图会把半成品冻结成「标准」，
> 且追求「整齐好看」正是 v1.13 抓到的目的论漂移陷阱（美 ≠ 证据）。详见
> docs/tufe-research-retrospective-origin-vs-now.md（起源 vs 现状 + 蓝图错位分析）。

## 〇、本文档的定位

本文档不是约束清单（那已在 tufe-design-constraints.md v0.4 + supplement v0.5 中完成），
也不是实验报告（那在 v12/v13 系列代码和结果中）。

本文档是一套**生成性规范**：给定此规范 + 任意物理基底，可以构造出一个
符合 TUF-E 约束的自学习智能体。

核心主张：**自学习 = 在耗散约束下，通过异构感知的交叉验证，积累经实践检验的自维持知识结构。**

[ME-C4] 以下所有内容均为 AI 综合推断，置信度标注于各节。

---

## 一、基本类型系统

设计语言定义五种基本类型。所有可表达的概念都是这些类型的组合。

### 1.1 Observer（观察者）

```
Observer := {
    id:           Identity,        -- 唯一标识
    type:         DiscretizationMethod,  -- 离散化方法（不可与其他观察者相同）
    params:       ProjectionParams,      -- C1投影元数据
    bandwidth:    Float,                 -- P4内部观测带宽预算
}
```

**不变量：**
- 系统中必须存在 ≥2 个 Observer，且 type 互不相同 (PC-1, C3.2)
- Observer 不可观察自身输出 (C5-B1)
- 每个 Observer 的离散化函数是私有的，不共享 (C3.2独立性)

### 1.2 Pattern（模式）

```
Pattern := {
    bucket:       BucketId,        -- 离散化后的桶标识 (b00-b15或更高粒度)
    observer_id:  Identity,        -- 产生此模式的观察者
    timestamp:    CycleIndex,      -- 产生的工作循环编号
    confidence:   Float ∈ [0,1],   -- 感知置信度
}
```

**不变量：**
- Pattern 永远携带来源 Observer 信息 (C1.1)
- 不存在"原始数据"，所有输入已经是 Pattern (C1.2)
- 桶的粒度是可调参数，但必须在同一 Observer 内一致 (PC-2)

### 1.3 Pivot（枢机）

```
Pivot := {
    in_pattern:   Pattern,         -- 输入状态模式（端）
    transform:    TransformId,     -- 因果变换标识（故）
    out_pattern:  Pattern,         -- 输出状态模式
    strength:     Float ∈ [0,1],   -- 当前强度（耗散变量）
    cycle_completions: Int ≥ 0,    -- 完整工作循环计数（时间层）
    confirmers:   Set[ObserverId], -- 确认此枢机的观察者集合
    birth_cycle:  CycleIndex,      -- 首次确认的工作循环
}
```

**三层完整性检查：**
- 结构层：in_pattern + transform + out_pattern 三者缺一不可
- 时间层：cycle_completions ≥ 1 才计入可行空间 (PC-5)
- 功能层：strength > death_threshold 才存活 (P1, PC-3)

**不可再分性判定：**
- 若 Pivot.transform 可分解为两个子变换 T1∘T2，且各自独立形成有效枢机，则原 Pivot 不是真正的枢机
- 实践中用操作空间的原子操作集作为不可再分的代理判据 (PR-1)

### 1.4 KnowledgeNetwork（知识网络）

```
KnowledgeNetwork := {
    pivots:       Map[PivotKey, Pivot],    -- 已确认枢机集合
    candidates:   Map[PivotKey, Candidate], -- 待确认候选
    topology:     BettiNumbers,            -- 当前拓扑不变量 (b0, b1)
    raf_status:   RAFClosureResult,        -- RAF闭合检验结果
}
```

**不变量：**
- 网络是耗散结构：所有 Pivot.strength 持续衰减 (P1, PC-3)
- 网络的自维持能力由 RAF 闭合度衡量 (PC-6)
- 拓扑变化(Betti数改变) = 潜在的学习事件 (BF-C1, BF-C6)

### 1.5 WorkCycle（工作循环）

```
WorkCycle := {
    index:        CycleIndex,
    observation:  Map[ObserverId, Pattern],  -- 各观察者的感知
    action:       TransformId,              -- 执行的变换
    outcome:      Map[ObserverId, Pattern],  -- 各观察者的结果感知
    viability_delta: Float,                  -- 可行空间变化量
}
```

**不变量：**
- 每个 WorkCycle 包含完整的 "感知→行动→结果感知" 序列
- viability_delta 的计算基于可行空间体积变化，不是外部奖励 (FS-C1, FS-C2)
- WorkCycle 是时间层的基本单位 (枢机的刹那)

---

## 二、核心运算

设计语言定义六种基本运算。所有系统行为都是这些运算的组合。

### 2.1 Perceive（感知）

```
Perceive(observer: Observer, raw_input: Any) → Pattern
```

- 将环境输入通过 Observer.type 离散化为 Pattern
- 消耗 Observer.bandwidth 预算 (P4)
- 产出携带 projection_id 和 confidence (C1)

### 2.2 Propose（提议）

```
Propose(network: KnowledgeNetwork, pivot: Pivot, observer: Observer) → Bool
```

- 将 Observer 发现的枢机三元组提交到共识池
- 如果该三元组的独立确认者数 ≥ min_observers 且平均强度 ≥ min_strength，则升级为 confirmed
- 这是 C3 共识机制的操作化定义

### 2.3 Decay（衰减）

```
Decay(network: KnowledgeNetwork, rate: Float) → Int  -- 返回过期数量
```

- 对所有 confirmed Pivot 执行 strength -= rate
- strength < min_strength 的枢机被移除
- 这是 P1 耗散约束的操作化定义
- rate 的有效区间由参数扫描确定：≤0.01稳定，≥0.05崩溃 (VC-2, PC-3)

### 2.4 Execute（执行）

```
Execute(agent: Agent, cycle: WorkCycle) → Outcome
```

- 根据当前 KnowledgeNetwork 选择变换
- 如果有匹配的已确认枢机(in_pattern匹配当前感知)，优先使用枢机引导的行为
- 否则按探索策略随机选择
- 记录完整 WorkCycle

### 2.5 Validate（验证）

```
Validate(network: KnowledgeNetwork) → ValidationReport
```

四层验证（顺序执行，任一失败则标记）：
1. **拓扑验证**: Betti数是否发生变化？(BF-C1)
2. **功能验证**: 变化前后 viability_delta 是否改善？(FS-C2)
3. **统计验证**: 置换检验 p-value < 0.05？(排除随机伪影)
4. **自维持验证**: RAF闭合率 ≥ 0.8？(PC-6)

全部通过 = 确认的顿悟(insight)。部分通过 = 候选顿悟。全失败 = 噪声。

### 2.6 Adapt（适应）

```
Adapt(agent: Agent, cycle: WorkCycle, feedback: Outcome) → ΔKnowledgeNetwork
```

- 根据 WorkCycle 的结果强化或削弱相关枢机
- 成功：strength += reinforcement_rate, cycle_completions += 1
- 失败：strength -= penalty_rate
- 更新 action_preferences（基于 viability_delta，不是 reward）
- 这是学习的操作化定义：**学习 = 知识网络的结构性调整，使可行空间得以维持或扩展**

---

## 三、系统组装规则

给定上述类型和运算，一个合法的 TUF-E 自学习智能体必须满足以下组装规则：

### R1: 最小观察者多样性

```
|{o.type : o ∈ system.observers}| ≥ 2
```

至少两种不同的离散化方法。这是共识有意义的前提。(PC-1, C3.2)

### R2: 耗散平衡

```
∃ rate ∈ (0, critical_rate) : 
    mean(pivot_creation_rate) > mean(pivot_decay_rate(rate))
```

枢机创建速率必须超过衰减速率，否则知识网络必然消亡。
critical_rate ≈ 0.015（来自VC-2的参数扫描）。(P1, PC-3)

### R3: 共识门控

```
∀ p ∈ network.confirmed : |p.confirmers| ≥ min_observers
```

没有经过跨观察者验证的知识不可进入 confirmed 集合。(C3, PC-4)

### R4: 时间层完整性

```
∀ p ∈ feasible_space_contribution : p.cycle_completions ≥ 1
```

只有经历过完整工作循环的枢机才计入可行空间体积。(PC-5)

### R5: 层级分离

```
meta_layer_rules ∩ object_layer_modifiable = ∅
```

元层规则不可被对象层修改。对象层的自我评估必须标记为 META_DECLARATION。(C5)

### R6: 可行空间导向

```
learning_signal := Δ(feasible_space_volume) ≠ external_reward
```

学习信号是可行空间的变化量，不是外部奖励标量。(FS-C1, FS-C2)

### R7: 拓扑学习判据

```
learned(t, t') ⟺ topology(KN_t) ≠ topology(KN_t') ∧ validation_passed(t')
```

"学到了新东西" = 知识网络拓扑发生了变化 AND 通过了四层验证。
参数收敛但拓扑不变 = 没有学到新东西。(BF-C6)

---

## 四、与现有范式的区分定理

以下命题区分 TUF-E 设计语言与三种主流范式。

### 定理 D1: TUF-E ≠ RL

RL 的学习目标是 max E[Σ γ^t r_t]。
TUF-E 的学习目标是 maintain {KN : RAF(KN) ≥ θ ∧ volume(KN) ≥ V_min}。

**区别:** RL优化标量信号的累积；TUF-E维持结构化知识的自持性。
当 reward hacking 使可行空间缩小时，RL认为在进步，TUF-E认为在退化。(FS-C4)

### 定理 D2: TUF-E ≠ LLM微调

LLM微调更新 P(token|context) 的参数分布。
TUF-E更新 KnowledgeNetwork 的拓扑结构。

**区别:** LLM的知识是隐式概率分布，不可直接读取或验证；
TUF-E的知识是显式枢机网络，每条知识都有来源、强度、验证状态。
LLM不会因不使用而遗忘；TUF-E的衰减是功能性筛选。(P1, PC-3)

### 定理 D3: TUF-E ≠ 符号AI

符号AI的知识是人工编码的逻辑规则。
TUF-E的知识是从交互中涌现的耗散结构。

**区别:** 符号AI的规则不会自主重组；TUF-E的枢机网络会通过相变自发重构。
符号AI没有耗散维持成本；TUF-E的知识需要持续能量投入。(P1, BF-C1)

---

## 五、开放问题与未决事项

### OQ-1: 自适应粒度机制
当前桶大小是固定参数。v1.3食堂域暴露了粒度不匹配的问题(PC-2)。
需要一种机制让 Observer 根据环境复杂度自动调整离散化粒度。
候选方向: 信息瓶颈理论(Tishby)、自适应直方图均衡化。

### OQ-2: 动态共识阈值
当前 min_observers 固定为2。PC-4指出应随学习阶段调整。
需要形式化"何时收紧/放宽共识标准"的判据。
候选方向: 知识网络成熟度指标(RAF闭合度+Betti稳定性)。

### OQ-3: 多智能体 vs 单智能体
v1.2用三个Agent模拟三个Observer。v1.3也沿用此设计。
但真实场景中可能是单个Agent内部的多个感知通道。
需要明确: Observer的物理载体是独立Agent还是内部模块？
这影响通信成本和独立性保证。

### OQ-4: 范畴论形式化的完成度
v0.4提到用对称幺半范畴或operad代数形式化枢机(Q0.3)。
当前设计语言用的是工程化的类型系统，尚未完成范畴论映射。
需要决定: 是否需要完整的范畴论语义？还是当前的类型系统已经足够指导实现？

### OQ-5: 从算术域到通用域的迁移协议
v1.3证明直接迁移离散化策略会失败。
需要一个通用的"域适配协议"，描述如何将TUF-E设计语言实例化到新领域。
关键步骤: (1)识别域的因果变换原子集 (2)设计域特异的Observer离散化 (3)校准衰减率和工作循环长度

---

## 六、版本历史

- v0.1 (2026-10-18): 初版。基于v1.2实验验证(PC-1~PC-6)和v0.4/v0.5约束文档提炼。
  包含: 5种基本类型、6种核心运算、7条组装规则、3条区分定理、5个开放问题。
- v0.1 状态标注 (2026-10-22, 未改正文): 加顶部诚实横幅。R7(拓扑判据)被 Betti churn 取代→学习信号改 OC-9;
  R3(共识)降为域受限(Ising 负效应); 扛事机制(SC-1/SC-2 因果记忆寻址+OC-12+F1/F2)在 v0.3 非本文档。
  声明 **v0.3＝当前活蓝图, v0.1＝历史起点**; 完整重写登记为待触发里程碑(触发=F1/F2 攻克+可持续结晶跑通)。
  据"订正不删除"纪律保留原文, 只加标注。
