# TUF-E Pivot 结构考古分析: v0.4 → v0.5+

## 一、丢失了什么？逐机制映射

| v0.4 结构 | v0.5+ 对应物 | 丢失的结构属性 |
|---|---|---|
| `HeterogeneousObserver.discretize_for_pivot()` — 三种观察者各自将同一数值离散化为不同模式串 | 不存在。v0.5 的 observation 是统一的 dict（inventory/personnel/equipment），所有 agent 看到相同特征空间 | **感知异构性**：同一现实→多个不可通约的符号描述 |
| `private_pivots` — `(in_pat, op_id, out_pat)` 三元组，强度随成功累积 | `pivot_library` — `(action, success_bool, reward_tier)` 三元组，仅记录统计 | **因果方向性**。v0.4 pivot 编码"输入状态→操作→输出状态"的变换关系；v0.5 pivot_library 只编码"动作→结果好坏"的相关性，丢失了状态变换语义 |
| `ConsensusKnowledgeStore.to_shared_key()` — 将私有感知空间的 pivot key 翻译到共享抽象桶空间 | 不存在。无翻译协议 | **跨视角可翻译性**。v0.4 的核心创新——不同观察者通过共享桶空间达成语义对齐——被完全移除 |
| C3 共识确认（≥2 独立观察者提案后才进入 confirmed） | 不存在。v0.5 无多 agent 共识机制；v0.8+ 的 consensus  operates on signal-level agreement，不是 pivot-level | **交叉验证的认识论约束**。知识不再需要独立视角的互证 |
| Pivot-guided action selection（40% 概率使用 confirmed pivots 选择操作） | `action_preferences` softmax 采样 + exploration rate | **知识驱动的行为选择**。v0.5 的选择基于偏好权重的强化学习，不查询任何结构化知识 |
| `decay(rate=0.003)` — 确认知识持续衰减，需主动维护 | 不存在。`action_preferences` 只做归一化，不做时间衰减 | **耗散维持**。知识不再是热力学对象，不会因缺乏使用而消亡 |

关键发现：v0.5 保留了 `pivot_library` 这个名称，但其结构已从"状态变换规则"退化为"动作-奖励关联表"。这不是演化，是同名异义。

## 二、为什么丢失？根因定位

直接原因是 **领域迁移时的表征不兼容**，但深层原因是 **固定动作集消除了 pivot 的存在理由**。

v0.4 的 pivot 之所以有意义，是因为操作空间是开放的：`op_id` 可以是任意算术运算，agent 需要发现"哪个操作在当前输入状态下产生期望输出"。Pivot 本质上是 **操作选择的依据**。

v0.5 将动作集硬编码为 6 个命名动作（inspect/order/menu_plan/intervene/staff_assign/record）。当动作空间封闭且语义固定时，"在什么状态下选什么动作"变成了一个标准的策略学习问题，不需要 `(in, op, out)` 三元组来表征——因为 `op` 不再是待发现的变量，而是枚举常量。

LLM 感知的引入（v0.6+）进一步固化了这一趋势：LLM 输出的信号已经是高层语义标签，天然与固定动作集对齐，无需经过离散化→翻译→共识的多步管道。

换言之：**pivot 是为开放操作空间设计的认知结构。当操作空间关闭时，它在架构上变成了冗余。**

## 三、食堂领域中保留 pivot 的结构等价物

如果 v0.5 要保留 v0.4 的 pivot 结构，其食堂域等价物应为：

`(situation_pattern, intervention_type, outcome_pattern)`

例如：`("库存低_人员充足", "order", "库存恢复_成本可控")`

这要求：
- **situation_pattern** 来自异构观察者的私有离散化（如一个 agent 按数量阈值分桶，另一个按品类组合模式分类，第三个按时段趋势标记）
- **intervention_type** 不是从固定集合中选，而是在参数空间中探索（如 order 的数量、时机、供应商组合构成连续操作空间）
- **outcome_pattern** 同样是观察者私有的结果描述，经 `to_shared_key()` 翻译后才能达成共识
- 多个 agent 对同一 situation→intervention→outcome 链的独立观察，经翻译后交叉验证

核心区别：v0.5 的实际实现中，动作是离散的、预定义的、语义透明的。Pivot 结构要求动作至少部分地是 **待发现的变换**，而非 **待选择的选项**。

## 四、可恢复性评估

**当前架构内不可恢复。** 原因：

1. `action_preferences` 的 softmax 选择机制与 pivot-guided 选择在数学上不兼容——前者是概率分布上的采样，后者是条件匹配后的确定性映射。两者可以混合（如 v0.4 的 40% 概率），但前提是存在可匹配的 pivot 结构。
2. v0.5+ 没有私有/共享感知空间的区分。恢复 pivot 需要重新引入 `HeterogeneousObserver` 和 `to_shared_key()` 翻译协议，这不是参数调整，是架构组件的添加。
3. 固定动作集本身需要被重新打开。只要动作是 6 个枚举值，pivot 的 `op` 维度就没有信息量。需要将动作参数化（如 order 变为 `(order, quantity_range, timing_policy, supplier_strategy)`），使操作空间具有足够的内部结构供 pivot 编码。

**恢复所需的最小架构变更：**
- 重新引入异构观察者 + 翻译协议
- 将动作从枚举改为参数化操作空间
- 在 `choose_action` 中增加 pivot-matching 分支
- 恢复耗散衰减机制

这不是"加一个模块"的工作量，而是对 v0.5 以来整个认知架构的结构性修订。Pivot 的丢失不是功能裁剪，是范式切换的副产品。
