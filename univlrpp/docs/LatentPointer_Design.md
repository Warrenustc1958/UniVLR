# UniVLR++：latent 软定位 + 高分辨率重看（Latent Pointer Glimpse）设计稿

日期：2026-11-12。状态：**方案，未实现、未验证**。下文的超参数都是起点建议，不是调好的配置。

## 0. 一句话

latent 不再承载视觉内容，只回答“看哪里”。指针从一次 prefill 中读出，用完后从上下文里删掉。答案只能通过重新编码的高分辨率子图拿到新信息，所以指针的作用可以直接检验：指针错了，子图就错了，分数应该明显下降。

依据（已核验）：
- latent 承载内容这条路走不通：oracle latent 注入不涨分；P0b 写入的内容与辅助图余弦 0.520，但只经 slots 看图的 V* 仍是 54.5。
- 收益来自“定位后重看”：同一张 GT crop，放在 latent 段 V* 84.7，作为 user 图像 95.2；E3 中低分辨率 crop 重新编码 V* +13.1。

---

## 1. 序列结构

### 1.1 主上下文（teacher 和 student 完全相同，只有 crop 的来源不同）

**2026-11-12 修订（已核验）**：布局改为原图 → 放大视图 → 问题 → 作答。原稿把 crop 放在问题之后，是为了复用"原图 + 问题"这段前缀；但实测这样上限会从 92.1 降到 81.7（V\*，GT crop）。原布局同样能复用前缀（共享 system + 原图，这是最贵的部分），计算量与"crop 后复述问题"持平，而且更简单。

```text
<|im_start|>system ...<|im_end|>
<|im_start|>user <vision_start> 原图 I 的 token <vision_end>               ← 共享前缀 P_I（长度 c0），只算一次、28 层
                 <vision_start> crop 的 token <vision_end>                  ← 作答前向追加
                 The second image is a zoomed-in view ... + 问题 q <|im_end|>
<|im_start|>assistant 答案（保留官方完整格式）
```

- teacher：crop = GT bbox 裁出的子图。student：crop = 指针选出的子图。
- **两者使用同一个裁剪函数**（同样的外扩规则、最小尺寸和 token 预算），避免把“裁剪方式不同”混进蒸馏信号。

| 布局（V\*，Qwen-Instruct 作答） | 只给原图 | GT crop | 指针 crop |
|---|---:|---:|---:|
| **原图 → crop → 问题（采用）** | 75.4 | **92.1** | **86.9** |
| 原图 → 问题 → crop | 75.4 | 81.7 | 80.6 |
| 原图 → 问题 → crop → 再说一遍问题 | 75.4 | 91.6 | 85.9 |

### 1.2 指针前向与作答前向

```text
指针前向：[system, user: 原图 I] ──28 层──→ cache 到 c0（原图结束处）
          追加 [问题 q, 指针 slot]，只算第 0–13 层；在第 14 层读出定位分布 π
          cache.crop(c0)：问题和 slot 的 KV 一起丢掉
作答前向：在 c0 的 cache 后追加 [crop, 说明 + 问题, assistant]，28 层
```

- 主上下文里没有任何 latent token。截断后 crop 段从位置 c0 开始编号，与"指针从未存在"完全一样（U5：logits 与无指针路径逐位相同）。
- student 的主上下文与 teacher 逐 token、逐位置相同，唯一的差别是 crop 的像素。
- 额外开销只有问题 token 再算一遍（几十个 token），且指针前向里的问题和 slot 只需 14 层。
- C 阶段的多个候选 crop 共享原图前缀，每个分支追加"crop + 问题"。

---

## 2. 逐项回答实现细节

### 2.1 latent 怎么输入

- **K = 1** 个指针 slot。Visual-CoT 每个样本只有一个 bbox，K > 1 学不出不同的区域。
- slot 的 layer-0 输入是一个**可学习 embedding** `e_ptr`（3584 维），不注入任何视觉特征，直接作为 `inputs_embeds` 传入，不占用词表 id。初值取词表 embedding 的均值。
- 位置：放在问题之后（同一个 user 回合内，`<|im_end|>` 之前）；文本型位置，M-RoPE 三个维度都取前缀最后位置 + 1。G3″ 显示这个位置上未训练的占位 slot 已能定位（第 14 层头 8、11、7、15）。
- 可训练参数（只作用于 slot）：
  - `e_ptr`；
  - **slot-only LoRA**：只在 slot 所在位置生效，放在第 1…L 层（q/k/v/o，rank 16，α = 32）；
  - 指针 head。
- slot-only LoRA 的好处：slot 位于前缀之后，因果注意力下前缀 token 看不到它，所以**原图和问题的 hidden 与 KV 一个比特都不变**。reader 不受影响，前缀 cache 可以在 teacher、student 和多个候选 crop 之间共享。

### 2.2 指针 head 与定位分布

记第 L 层 slot 的 hidden 为 `h`，原图第 j 个合并视觉 token 的 hidden 为 `v_j`（j ∈ Ω_I，只在原图 token 上取）：

**2026-11-12 实现版（ptrB.py）**：在 Qwen 自带的第 14 层注意力上加零初始化修正项，使第 0 步等于不训练的指针。

```
p_nat_j = mean_{h∈{8,11,7,15}} softmax_j(q_h(h_slot) · k_h(v_j) / sqrt(128))     Qwen 原生第 14 层注意力（含 M-RoPE）
δ_j     = (W_q · LN(h)) · (W_k · LN(v_j)) / 16,   W_q 零初始化
log π   = log_softmax_j( log p_nat_j / τ + δ_j ),  τ 可学习，初值 1
```

- W_q、W_k 各为 3584 × 256，可适应任意分辨率的网格。
- q_h 来自 slot（经 slot-only LoRA），k_h 来自冻结的原图 token，二者都按 Qwen 自身的投影和 M-RoPE 计算。
- `v_j` 来自前缀前向，需要用 hook 存下第 L 层的原图 hidden（KV cache 里只有 K/V，没有 hidden）。
- softmax 只在原图 token 上做，`<vision_start>`、系统提示首 token 这类常见 sink 本来就不在候选里。高范数离群 token 的影响留作消融。
- L 默认取 14（与 UniVLR 的对齐层一致），扫描 {8, 14, 20, 28}。CVRR 报告 Qwen2.5-VL 文本读图大约在第 20 层前结束，这是选 L ≤ 20 的参考。
- 层号约定：`hidden_states[L]` 表示第 L 层的输出（`hidden_states[0]` 是 embedding）。

### 2.3 从分布得到 crop

1. 网格坐标：合并 token 对应缩放后图像中 28 × 28 px 的块，按缩放比例映射回原图像素坐标。
2. 按 π 从大到小取最少的格子，使累计质量 ≥ ρ（默认 0.6）；取其中最大的 8 连通分量，求外接框。
3. 外扩规则与 GT crop 完全相同（沿用 Visual-CoT 筛选代码：每边外扩 bbox 的宽或高，最小 56 px，裁到图像边界内）。
4. **从原图的原始像素上裁**，而不是从 reader 视图上裁，再缩放到 crop 预算（建议 256–1280 个合并 token）。小区域会被放大，这就是“zoom”。
5. 这一步不可导，指针的梯度来自 §4 的定位损失和视角信用。

消融：从 reader 视图裁剪（E3 的设置）vs. 从原图像素裁剪；ρ ∈ {0.4, 0.6, 0.8}；最大连通分量 vs. 加权外接框。

### 2.4 答案要不要删掉 latent 的 KV

**默认删除。** 原因：
1. **teacher 和 student 的上下文完全一致**，蒸馏信号只反映“视角”的差别。
2. **可证伪**：答案只能通过 crop 受指针影响，不存在“指针 slot 本身被当作内容载体”的旁路。这正是 Monet、UniVLR 类方法被 CVRR、Beyond Visual Memory 批评的地方（后者发现只保留边界标记就能保住 78–100% 的增益）。
3. **cache 复用是精确的**，不需要特殊的注意力掩码。

实现（已验证，transformers 4.54）：prefill [system, 原图]（28 层）→ 追加问题和 slot（只算到第 14 层，用 pre-hook 抛出 Stop 提前结束）→ 读出 π → `DynamicCache.crop(c0)` 截回原图结束处。前 14 层多出的问题和 slot KV 被截掉，后 14 层本来就只到 c0（U5：截断后两段长度都是 c0）。

消融：保留指针 KV，让 crop 段和答案都能注意到它。

### 2.5 latent 如何“自回归”

- **单次 glimpse 内不做 token 级自回归。** 不像 UniVLR 那样把 hidden 回填到下一个位置。方向 A 的 L0 已经发现，那种 writer 根本不看自己之前写的 latent（oracle 前缀注入后余弦 0.996–0.999），自回归只增加串行成本。
- **自回归放在 glimpse 级（多步时）：**
  ```text
  P → 旁路 ptr_1 → 删除 → 追加 crop_1
    → 旁路 ptr_2（能看到 I、q、crop_1）→ 删除 → 追加 crop_2
    → … → 答案
  ```
  第 t+1 个指针看不到第 t 个指针的 KV（已删除），但能看到第 t 个指针的结果 crop_t。每一步都有新的像素进来，这比回填 hidden 更有意义。
- 指针始终在**原图网格**上打分（缓存的第 L 层原图 hidden）。
- 默认 T = 1。T = 2 需要多区域监督数据，现在没有，先不做。停止机制（学习何时不再看）留到以后。

### 2.6 高分辨率图如何复用第一遍 prefill

- 原图只过一次 ViT、一次语言模型 prefill。crop 段单独过 ViT（Qwen2.5-VL 的 ViT 对每张图独立编码，与原图无交互），再以 P 的 cache 为 past 增量 prefill。
- **位置编码必须与整段 prefill 一致。** 不要手写 M-RoPE 的偏移：先拼出完整的 `input_ids` 和两张图的 `image_grid_thw`，调用模型自带的 rope index 函数（Qwen2.5-VL 的 `get_rope_index`）计算整段位置，再切出 crop 段那一部分。解码时同步更新 `rope_deltas`。
- 第二遍的额外开销：一次 ViT(crop) + crop token 和答案 token 的语言模型前向。原图不重新编码。
- teacher 打分多个候选 crop 时（§4.3），前缀 cache 只算一次，每个候选复制或扩展 batch 维度后各自追加 crop 段。

### 2.7 单元测试（已全部通过，2026-11-12，Qwen2.5-VL-7B-Instruct）

| 测试 | 结果 |
|---|---|
| U1 增量 prefill（crop 单独过 ViT）== 整段 prefill | 与自身整段前向 KL 0；与 HF 官方前向 KL 1.1e-5，argmax 100% |
| U2 指针旁路 + cache 截断 == 无指针路径 | logits 最大差 0.0 |
| U3 slot-only LoRA 开关前后前缀 KV | 最大差 0.0，slot hidden 有变化 |
| U4 追加 crop 后前缀 M-RoPE 位置 | 完全相同 |
| **U5 新截断点**：原图 28 层 → 问题+slot 到 14 层 → crop(c0) → 追加 crop+问题，与整段前向 | KL 0，argmax 100%；与无指针路径最大差 0.0 |
| **U6** 训练路径（一次前向到 14 层）与部署路径读出的 π | KL 0.0013，概率最大差 0.005，argmax 相同（bf16 分块差异） |
| **U7** 梯度只到指针参数 | 底座 0 个参数有梯度，指针参数全部有梯度 |
| **U8** slot 前向（LoRA 开）不改前缀 KV | 最大差 0.0 |

注意：`vlmeval/vlm/__init__.py` 在导入时调用 `torch.set_grad_enabled(False)`，训练代码必须显式打开梯度。

---

## 3. 推理流程

```text
1. ViT(I)，按官方评测的原图预算（5120 token）
2. prefill [system, user: I]（28 层），缓存 KV；hook 存下第 14 层原图 hidden
3. 追加 [q, 指针 slot]，只算第 0–13 层（slot-only LoRA 只在 slot 上开启）→ 第 14 层读出 π
4. cache.crop(c0)，丢掉 q 和 slot 的 KV
5. 由 π 取峰值连通区域 → 外扩 2 倍（最小 224 px）→ 从原图原始像素裁剪 → ViT(crop)
6. 追加 [crop, 说明 + q, <|im_end|>, assistant]，按整段 get_rope_index 计算位置
7. 贪心解码答案
```

---

## 4. 训练阶段与起点权重

### 4.0 数据准备（2026-11-12 实际执行）

- cpfs 上没有原版 Visual-CoT 标注，bbox 由 Monet 的 crop 在原图中模板匹配反推（TM_CCOEFF_NORMED，Visual-CoT ≥ 0.9；抽查 300/300 ≥ 0.9、260/300 ≥ 0.98）。Zebra visual_search 只保留原尺寸精确匹配（≥ 0.95，约 60%）。
- 数据量对齐 Monet / UniVLR-Stage1 的 Visual-CoT（118,561 条；LVR 复现常用 6 万条）。构建结果：训练 118,988 条（Visual-CoT 117,275 + Zebra visual_search 1,713），留出 492 条（P0 留出集，8 条匹配失败）。与 V\*、HR4K、HR8K、MME-RW-Lite 共 3,710 张图做 32×32 哈希去重，无重复。bbox 面积占原图 p10 / p50 / p90 = 0.4% / 6.9% / 43%。不做 5% 面积过滤。

原稿内容（保留作参考）：

- bbox 来源：**原版 Visual-CoT 的标注里已经有 bbox**（图像路径中 `###` 后面的 JSON，`screen_full.py` 已经能解析）。Monet 的 json 只有 crop，没有 bbox，不用它。
- 已知：404,120 条标注，其中 44,013 条缺图、3 张坏图。指针训练**不要沿用筛选时 bbox 面积 < 5% 的过滤**，大区域也要学。
- 按图像 id 和来源划出 Visual-CoT dev 集。与 V*、HR-Bench、MME-RealWorld 的图像去重；V* 的图像来源需单独核查。
- 少量没有 bbox 的通用 VQA 样本，GT box 设为整图，避免模型学成“永远放大”。

### 4.1 Stage A：会读 crop 的 reader（是否需要取决于 G1′）

- 起点：Qwen2.5-VL-7B-Instruct。
- 格式：§1.1，crop = GT crop。
- 数据混合：约 60% GT crop、约 30% 抖动 crop（与 GT 的 IoU 在 0.3–0.9 之间）、约 10% 整图作为 crop。目的是让 reader 能用不完美的 crop，并在 crop 没用时退回原图。
- **保留官方完整的 assistant 答案**，不做 P0b 那样的清洗（P0b 在 MME 上有 33% 的答案无法解析）。不屏蔽原图。
- 参数：LoRA（rank 64，全部线性层，ViT 冻结）或全量微调语言模型，二选一，先用 LoRA。
- 输出 ckpt A，同时作为 reader 和 teacher（冻结）。
- 如果 G1′ 显示 Qwen-Instruct 在这个格式下已经能充分利用 GT crop，A 可以只做轻量的格式适配，甚至跳过。

### 4.2 Stage B：指针热启动（只训练指针）

- **2026-11-12 实际配置（ptrB.py，dsw3 16 卡）**：A 阶段跳过（G1′：Qwen-Instruct 不训练就能读 GT crop，V\* 92.1），起点 = Qwen2.5-VL-7B-Instruct，**全部冻结**。可训练：`e_ptr`、slot-only LoRA r16（第 0–13 层 q/k/v/o + 第 14 层 q）、W_q/W_k/LN、τ，共 701 万参数。损失 CE(m\*, π)。全局 batch 64，1,859 步，LoRA lr 2e-4、head lr 1e-3，余弦衰减，warmup 3%。每步 1.45 秒，显存 17.6 GB/卡。
- 原稿：起点 ckpt A，**全部冻结**。只训练 `e_ptr`、slot-only LoRA 和指针 head。
- 损失：`L_ptr = KL(m* ‖ π)`，其中 `m*_j` 是第 j 个格子与 GT bbox 的重叠面积，归一化成分布。可选：UniVLR 式的第 14 层对齐 ViT(GT crop)，作为辅助信号，放进消融。
- 不需要答案前向：reader 冻结，B 阶段只需要前缀和指针 slot 的前 L 层前向。可以把 `model.layers` 截到前 L 层来省一半计算。backward 只经过 slot 路径。
- 因为 reader 不训练，这一阶段不需要 scheduled sampling。

### 4.3 Stage C：特权视角的候选重加权（on-policy 指的是视角）

- 起点：指针来自 ckpt B；reader 和 teacher 都是冻结的 ckpt A。
- 每个样本：
  1. student 前向得到 π；
  2. 构造 m = 4 个候选 box：贪心 box（ρ = 0.6）、贪心 box（ρ = 0.8）、GT box、抖动后的 GT box；
  3. 前缀 cache 算一次，分支追加 m 个 crop，teacher 对每个候选打分：
     `d_c = −(1/|y|) Σ_t log T(y_t | P, crop_c, y_<t)`，或者与“看 GT crop”时答案分布的 KL；
  4. `w = softmax(−d / τ_v)`，τ_v 按同一样本内 d 的标准差归一化；
  5. 目标分布 `m̃ = Σ_c w_c · m_{b_c}`，指针损失 `L_view = KL(m̃ ‖ π)`。
- 和 B 的区别：当某个预测的 box 比 GT box 更有用时（GT 不一定是唯一最好的视角），指针会学向它。这是 B 做不到的。
- **如实命名**：这是候选重加权，接近 reward-weighted regression。没有策略梯度，也没有对答案 token 做 on-policy 采样。“on-policy”只指 crop 来自 student 自己的指针。
- 可选的 reader 适配：单独加一个 reader LoRA ω，用 `KL(T(·|P, GT crop) ‖ R_ω(·|P, 自选 crop))` 训练，让 reader 适应自己选出的 crop。作为消融，默认关闭。
- 数据：从 B 的训练集里取 2–3 万条。每个样本要多做 m 次 teacher 前向（前缀共享）。

---

## 5. 开工前的门槛（不需要训练）

| 门槛 | 内容 | 通过标准 |
|---|---|---|
| G1′ 上限 | Qwen-Instruct 用 §1.1 的格式，在 Visual-CoT dev 和 V* 上比较：不给 crop / GT crop / 整图作为第二张图（相同 token 预算）/ 随机 crop / 中心 crop。HR 和 MME 没有 bbox，用“答案 oracle 网格 crop”（在候选网格 crop 中按 gold 答案挑最好的）作为**偏乐观**的上限 | GT crop 明显高于“整图第二张图”，才能说明收益来自位置而不只是多给 token |
| G2′ 信号平滑 | IoU 从 1 扫到 0，teacher 的 gold logp 是否平滑单调 | 单调且噪声小，Stage C 的打分才可用 |
| G0′ 解释验证 | 官方 Stage1 的 latent 置零或换成别的样本，测 HR4K 和 MME | 基本不变，支持“UniVLR 的 latent 主要是训练信号” |
| 单元测试 | §2.7 的三项 | 全部通过 |

G3′（P0b 的 slot 注意力与 bbox 的重合度）在新方案里不再决定起点（起点改为 Qwen-Instruct 或 ckpt A），降为诊断项。

---

### 5.1 门槛实验结果（2026-11-12，不训练，作答模型 Qwen2.5-VL-7B-Instruct）

| 门槛 | 结果 | 结论 |
|---|---|---|
| G1′ 上限（V\*，原布局） | 只给原图 75.4，加整图 80.6，GT crop 92.1（+16.8 [+11.5, +22.5]） | 通过；扣掉"多一张图"后视角特权约 11.6 分 |
| G1″ 对照（V\*，crop 在问题后的布局） | GT 81.7；token 数相同的整图 77.5，随机 crop 77.5，中心 crop 78.0 | 收益来自位置，不只是多给 token |
| G2′ 信号平滑 | IoU 1→0.7→0.5→0.3→0：正确选项概率 0.870→0.865→0.825→0.752→0.763 | 到 0.3 平滑下降，C 阶段打分可用 |
| G3″ 未训练 query 能否定位 | 问题后 1 个占位 slot，第 14 层头 8、11、7、15：随机的 19.5 倍（逐题比值平均） | B 阶段可直接从 Qwen-Instruct 起步；P0b 的 slot 换到新位置只剩 4 倍 |
| G4′ 不训练的端到端（P0b 第 14 层指针） | V\* 86.9（比只给原图 +11.5，比加整图 +6.3）；HR4K 76.1（+11.4 / +5.8，单目标 +16.5，多目标 +6.2） | 链路不训练已有效；多目标题是主要短板 |
| 多峰读出 | 两峰合并 79.6，两张 crop 80.1，单峰 80.6（新布局） | 不采用 |
| G0′ Stage1 | 官方评测（univlr 1 步）生成的 latent 段为空，回填 0 次 | Stage1 比 Qwen 高的分数与推理时 latent 无关 |
| 文本 bbox 基线 | 待公平版本（重提示 + 配对 CI）结果 | 进行中 |

## 6. 对照、消融与可证伪检验

**必须有的对照：**
1. 不给 crop。
2. 整图作为第二张图，token 数与 crop 相同：区分“看对了地方”和“多给了 token”。
3. 随机 crop、中心 crop。
4. GT crop：上限。
5. **文本 bbox 基线**：同样的数据，模型以文本输出坐标，再裁剪重看。这是最关键的对照。如果 latent 指针与它持平，latent 的贡献就只剩效率（少生成十几个坐标 token），而不是准确率。
6. 外部方法：ZoomEye、DeepEyes、Chain-of-Focus、mini-o3，按统一协议评测。

**可证伪检验：** 指针置零、打乱、换成别的样本的指针、固定为中心或整图。分数应该明显下降。

**消融：** 删除 vs. 保留指针 KV；L 的选择；ρ；crop 来源（原图像素 vs. reader 视图）；crop 预算；有无 Stage C；有无第 14 层对齐辅助；有无 reader 适配；排除高范数 token。

---

## 7. 风险

1. **文本 bbox 基线可能持平**，见 §6 第 5 条。
2. **撞车风险**：“先定位再放大”已有很多工作（ZoomEye、DeepEyes、Chain-of-Focus、mini-o3）。“用内部状态定位、不输出坐标”需要专门检索，没查之前不能声称是新的。glimpse 模型（Mnih et al., 2014）是概念源头，要如实引用。
3. **单区域局限**：HR 和 MME 中需要同时看多处的题，T = 1、K = 1 处理不了。
4. **领域偏移**：Visual-CoT 的 bbox 集中在文档、图表、自然图像几类，迁移到 HR 的 4K/8K 图像需要实测。
5. **推理型特权不归这个方法管**：maze 这类 canvas 里有原图读不出的中间过程，重看原图拿不到。渲染主线作为局限或分析章节保留。
