# AEQGA 函数拟合类

实现文件：`aeqga.py`。使用现有的 `qml` Conda 环境，无需安装新包。
已确认环境版本为 Python 3.11、NumPy 2.0.1、Qiskit 2.3.0、Qiskit Aer 0.17.2。

## 最小用法

```python
from aeqga import AEQGA

def function(x: list, p: list):
    # x 是一条样本的特征列表，p 是待求参数列表。
    return p[0] * x[0] + p[1]

model = AEQGA(
    function,
    p=[0.0, 0.0],                   # 初始参数，会作为候选解保留在初始种群中
    bounds=[(-5, 5), (-3, 3)],       # 与 p 一一对应的搜索范围，必填
    seed=7,
)

x = [-2, -1, 0, 1, 2]              # 也可以写成 [[-2], [-1], [0], [1], [2]]
y = [-5.7, -3.2, -0.7, 1.8, 4.3]
model.train(x, y, generations=150)

print(model.p)                      # 期望接近 [2.5, -0.7]，实际精度取决于搜索
print(model.get_p())                # 同样返回参数的副本
print(model.loss_)                  # 最佳拟合误差 SSE
print(model.predict([0, 1, 2]))
```

目标函数的形式由你定义；`p` 中每个元素都作为连续实数参数搜索。
`p` 的 getter 返回列表副本，修改该列表不会修改模型。调用 `train()` 返回模型本身。
再次调用 `train()` 会以当前最优参数作为一个初始候选，重新生成其余种群，并重置训练记录。

## 函数输入约定

默认 `vectorized=False`：对每条样本调用 `function(x_list, p_list)`，返回一个实数。
多输入特征示例：

```python
def function(x, p):
    return p[0] * x[0] + p[1] * x[1] + p[2]

# train 的 x 形状是 (样本数, 特征数)，每条样本内部的顺序由你约定。
x = [[1.0, 2.0], [2.0, 3.0], [4.0, 1.0]]
```

模型函数需要在给定参数范围内有定义。数值结果为 NaN/Inf 的候选会得到无穷大误差，
从而被淘汰；用户函数中的异常与输出形状错误会直接报告，避免静默掩盖实现错误。
模型函数应无副作用、输出确定，因为保留个体的误差会被缓存。

如果函数一次接收整批输入并返回所有预测值，设置 `vectorized=True`：

```python
import numpy as np

def batch_function(x_list, p_list):
    x = np.asarray(x_list)          # 此例传给 train 的 x 是一维列表
    return p_list[0] * x**2 + p_list[1]

model = AEQGA(batch_function, [0, 0], bounds=[(-3, 3), (-2, 2)],
              vectorized=True, seed=17)
```

批量模式保留 `train` 输入的维数：一维列表得到一维输入，二维样本列表得到二维输入。
批量函数必须返回形状为 `(样本数,)` 的列表或数组。数值计算较重时，建议使用批量模式。

## 训练控制与误差定义

构造参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `population_size` | 32 | 种群大小，必须是至少 8 的 2 的幂；少于 4 个候选的量子分组会补零到 4 个振幅 |
| `elite_count` | `population_size // 4`（默认 8） | 原样保留的优秀个体数，至少 1 |
| `local_count` | `population_size // 4`（默认 8） | 优秀解附近搜索生成的个体数，可以为 0 |
| `random_count` | `population_size // 2`（默认 16） | 全局随机初始化并经过量子操作的个体数，可以为 0 |
| `shots` | 1024 | 每条量子线路的测量次数 |
| `parallel_circuits` | 1 | 同一代一次执行最多合并的独立小电路数量，正整数 |
| `max_parallel_qubits` | None | 每个合并电路允许的活跃 qubit 上限，超出时自动分组；单个小电路已超限则报错 |
| `crossover_probability` | 0.5 | 执行双向受控 Y 旋转的概率 |
| `mutation_probability` | 0.5 | 执行单量子比特 X 旋转的概率 |
| `local_padding` | 0.5 | 优秀解最小/最大值区间两侧的扩展比例 |
| `min_local_width` | 0.01 | 局部区间最小宽度占全局范围的比例，扩展前计算 |
| `vectorized` | False | 是否一次计算整批输入 |
| `seed` | None | 固定后，同环境、同输入、同初始参数的结果可复现 |

三个数量必须加起来等于 `population_size`。不传（或传 `None`）时，分别按原来的
25%、25%、50% 取值；省略的数量不会自动吸收其他组的差额。例如：

```python
model = AEQGA(
    function, p=[0, 0], bounds=[(-5, 5), (-3, 3)],
    population_size=32,
    elite_count=5,     # 保留上代排名最好的 5 个
    local_count=11,    # 在优秀解附近生成 11 个
    random_count=16,   # 随机初始化 16 个，进入全局量子搜索
    seed=7,
)
```

局部数量大于保留数量时，按优劣顺序循环复制优秀个体；小于时取其中最好的部分。
局部搜索区间仍由所有保留个体决定。每个量子分组可以不是 2 的幂：编码时补零到
不小于 4 的最小 2 的幂，解码后只保留所请求的样本数；默认配置不需要补零，行为保持原样。
数量为 0 的搜索分组不会运行量子线路。若全部个体均保留，则种群只保留和排序，不再探索。

`train(x, y, generations=100, sigma=None, covariance=None,
target_loss=None, verbose=False)`：

- 默认最小化 `sum((prediction - y)**2)`，即 SSE。
- `sigma` 可以是正标量或每条数据的标准差；此时最小化
  `sum(((prediction - y) / sigma)**2)`。
- `covariance` 是对称正定的数据协方差矩阵；此时最小化
  `residual.T @ inverse(covariance) @ residual`。实现使用 Cholesky 分解和三角求解，
  不显式求逆。`sigma` 和 `covariance` 不能同时传。
- `target_loss` 是上述误差的绝对阈值；达到后提前结束。默认完成给定代数。
- `verbose=True` 输出第 0 代及每 10 代的最佳误差和参数。

```python
model.train(x, y, generations=200, sigma=0.03, verbose=True)
```

训练记录：`history_` 包含第 0 代和每代的 `generation`、`loss`、`p`（每代最佳解），
以及 `best_index`、`loss_kind` 和 `population`（该代全部候选）。
`population` 中每个候选保存 `p`（全部拟合参数）、`loss` 和 `chi2`。
传入 `sigma` 或 `covariance` 时 `loss_kind="chi2"`，`chi2` 等于 `loss`；
未加权拟合时 `loss_kind="sse"`、`chi2=None`，避免将未加权残差平方和误称为卡方。
所有参数均按代复制保存，包含保留的精英、局部与全局新个体，且包含最后一代；
无效数值候选的损失保留为 `inf`。种群按当前排列记录，未额外排序。

```python
generation = model.history_[10]
candidate = generation["population"][0]
print(candidate["p"], candidate["chi2"])
model.export_population_history(
    "output/population_history.csv",
    parameter_names=["H0_km_s_Mpc", "Omega_m", "M_mag"],  # 与模型参数数目和顺序一致
)
```

CSV 每行对应一个候选，包含 `generation`、`individual`、全部参数和 `chi2`（未加权时为 `sse`）。
`individual` 从 0 开始，是该代中的位置，并非跨代不变的个体身份；默认参数列名为 `p_0`、`p_1` 等。
1000 代、32 个体导出 `(1000+1)*32=32032` 行数据，包括初始化的第 0 代。
再次训练会重置记录；记录内存随代数、种群大小和参数数目增加。

`n_generations_`、`n_evaluations_`、`quantum_circuits_` 分别记录已完成代数、
候选参数的误差计算次数、逻辑小电路的采样数。
`executed_circuits_` 记录合并后实际执行的电路数；默认 `parallel_circuits=1` 时两者相同。
默认每代执行 `2 * len(p)` 条线路；每个参数的局部和全局部分使用独立线路。
关闭其中一组后，每代执行 `len(p)` 条；两组都关闭时为 0 条。
总误差计算次数为 `population_size + 实际代数 * (local_count + random_count)`。
所有新个体，包括最后一代新个体，都经过误差评价。

## 与论文对应的计算逻辑

依据本目录论文第 3 节、Algorithm 1/2、公式 (11)-(18)：

1. 经典端计算误差，默认保留最好的 25% 个体；复制它们得到 25% 副本；随机生成另外 50%。
   也可以通过 `elite_count`、`local_count`、`random_count` 分别调整数量。
2. 每个参数维度分别对副本和随机部分进行直接振幅编码 `x / ||x||`。
3. 以给定概率，对随机两个量子比特应用 `CRy(pi/2)`，交换控制和目标后再应用一次；
   以给定概率对一个量子比特应用 `Rx(pi/2)`。
4. 在 `AerSimulator(method="statevector")` 中实际执行带测量的线路，使用有限 shots
   得到 `counts`。这里的 statevector 是模拟方法，解码使用测量计数，没有直接读取精确振幅。
5. 局部部分使用 `lo + (hi - lo) * sqrt(count_i / shots)`；全局部分使用
   `lower + (upper - lower) * (count_i - min_count) / (max_count - min_count)`。
6. 原样保留个体和解码后的个体合并，再进行下一代搜索。

为了让算法适用于一般函数，本实现明确采用以下工程约定：

- 将用户给定的初始参数加入初始随机种群。
- 论文没有给出局部搜索盒的具体宽度计算方式；本实现使用优秀解的逐维最小/最大区间，
  加最小宽度和两侧扩展，最终限制在参数上下界内。两个旋钮见上表。
  默认保留全局宽度 1% 的局部搜索宽度，避免种群过早收缩；找到合适区域后可以减小
  `min_local_width` 做更细的搜索，但过小会增加停滞风险。
- 支持负参数，直接作为带符号的实振幅编码；全零向量用均匀态代替，因为零向量无法归一化。
- 对全局分支所有计数相同的情况，用区间内随机值处理公式分母为零的问题。
- 按论文 Algorithm 2，对恰好落在解码下界的数值在对应区间内重新随机生成。
- 缓存未改变的优秀个体误差；无效数值候选不足以组成优秀集合时，用有效优秀个体补齐。

这保留了 AEQGA 的核心操作，但不是作者源代码或论文实验数值的逐项复现。
算法不使用梯度下降、SciPy 优化器或经典最小二乘来修正最终参数。
搜索是随机启发式过程，参数范围、代数、种群和 shots 会影响精度，不能保证全局最优。
目标函数计算仍为经典计算，使用量子模拟不意味着比经典拟合更快。

Qiskit 接口参考：[QuantumCircuit 2.3](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.circuit.QuantumCircuit)、
[AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)。

## 在本地运行

```powershell
conda activate qml
python example_aeqga.py
python -m unittest -v test_aeqga
```

也可直接运行：

```powershell
conda run --no-capture-output -n qml python example_aeqga.py
conda run --no-capture-output -n qml python -m unittest -v test_aeqga
```

示例拟合带噪声的直线，真实参数是 `[2.5, -0.7]`。示例末尾的经典最小二乘仅用于独立比较，
不会参与 AEQGA 训练。测试覆盖实际 Aer 调用、振幅和门操作、最终一代评价、精英保留、
非线性拟合、批量/单样本接口、多特征、随机种子复现、加权误差、数值异常和输入校验。

本地验证：12 项测试通过。示例设置下得到参数约 `[2.50496, -0.70593]`，
SSE 约 `0.02646`；同一带噪声数据的经典最小二乘参考参数约为
`[2.50296, -0.69834]`。两者的区别来自随机搜索误差；参考计算没有参与搜索。

## 导出每代最优解折线图

训练后直接调用：

```python
model.export_history_plot("output/training_history.png")
```

默认分别绘制所有参数与误差的折线图。每个点取自该代 **误差最小的个体**；
参数值本身不一定单调变化，最优误差应不增加。默认不画初始化的第 0 代，
运行 1000 代对应图中 1–1000 代的 1000 个点，不进行平滑或抽样。

```python
model.export_history_plot(
    "output/h0_history.png",
    parameter_indices=[0],           # 只画 p[0]
    parameter_names=["H0", "Omega_m", "M"],  # 全部参数的名字，顺序与 p 相同
    include_loss=False,
    include_initial=False,
    title="Pantheon+SH0ES | AEQGA | 1000 generations",
    dpi=180,
)
```

方法返回导出文件的绝对 `Path`，自动创建父目录；扩展名可用 `.png`、`.svg` 等。
`include_initial=True` 可加入第 0 代，`loss_label` 可设置误差轴名称。
方法使用后台画布，不弹出窗口；没有训练记录时会抛出明确错误。

## Pantheon+ 数据集 1000 代运行

```powershell
conda run --no-capture-output -n qml python run_pantheon_aeqga.py
```

默认参数：1000 代、32 个体、每线路 1024 shots、随机种子 7。
可用 `--generations`、`--seed`、`--output` 改变代数、种子和输出目录；
`--population`、`--shots`、`--parallel-circuits`、`--max-parallel-qubits` 控制种群、测量和合并方式。

按本地数据附带的 `Pantheon+SH0ES_cosmosis_likelihood.py`，本次拟合使用：

- `zHD > 0.01 OR IS_CALIBRATOR == 1`，保留 1657 条光变曲线记录，包括 77 条校准记录。
  同一天体的重复测量按原始顺序保留，相关性通过完整协方差处理。
- 拟合量是 `m_b_corr`；校准记录的模型是 `CEPH_DIST + M`。
- 对非校准记录，平直 ΛCDM 的距离为
  `d_L = (1+zHEL) * (299792.458/H0) * integral_0^zHD dz / sqrt(Omega_m*(1+z)^3+1-Omega_m)`，
  模型星等为 `5*log10(d_L/Mpc)+25+M`，忽略辐射。
- 同时拟合 `p=[H0, Omega_m, M]`，初值 `[70, 0.3, -19.3]`，
  范围分别为 `[60,80]`、`[0.1,0.5]`、`[-19.6,-19.0]`。
  使用 SH0ES 校准来约束绝对距离尺度，避免把未校准超新星的星等简并当成 H0 测量。
- 使用完整的 `STAT+SYS` 协方差。原始所选矩阵最大非对称差约 `3e-8 mag^2`，
  按 `(C+C.T)/2` 处理文件小数精度引起的非对称；保留所有非对角关联。
  处理前的差异、源文件哈希和最终设置都写入结果 JSON。
- 训练使用 32 点 Gauss–Legendre 积分，并用独立复合 Simpson 积分检查精度；
  最终误差再用 64 点积分复算。训练后计算经典剖面最小二乘参考值，仅用于检查搜索差距，
  不参与初始化、训练或最终参数替换。

输出位于 `output/pantheon_aeqga/`：

| 文件 | 内容 |
|---|---|
| `h0_history.png` | 1000 代最优 H0 的折线图 |
| `h0_history.svg` | 同一图的可缩放矢量版本 |
| `all_parameters_and_loss.png` | H0、Omega_m、M 和 chi-square 的完整轨迹 |
| `history.csv` | 每一代最优参数与误差，共 1000 行数据 |
| `population_history.csv` | 每代全部候选的参数和卡方，含第 0 代，每行一个候选 |
| `summary.json` | 最终结果、设置、数据选择、版本、验证与参考解 |
| `execution_history.jsonl.gz` | 每代合并分组、物理/逻辑位置、测量位映射、门数、活跃 qubit 数和原始联合计数 |
| `last_generation.qpy` | 最后一代可执行的合并电路，可通过 `qiskit.qpy.load` 读取 |
| `last_generation_layout.json` | 最后一代电路分组与位置，和 QPY 文件逐条对应 |

这是一次训练中的 1000 代，不是 1000 次独立训练。曲线表示优化过程，不能作为后验分布或 H0 的置信区间。
本次三参数校准拟合与原论文固定校准后只拟合两个宇宙学参数的实验设置不同。
数据方法来源：[Pantheon+SH0ES 官方发布仓库](https://github.com/PantheonPlusSH0ES/DataRelease)。

新增绘图与宇宙学公式测试：`conda run -n qml python -m unittest -v test_history_plot`。

## 总种群与最终拟合结果对比

运行 `conda run --no-capture-output -n qml python run_population_sweep.py`，分别以
8、16、32、64 个体训练，默认每组 1000 代、1024 shots、seed=7，精英/局部/全局比例为 1:1:2。
每组均从相同初始参数独立训练。输出位于 `output/population_bit_sweep/`，其中
`final_parameters_vs_bits.png` 和 `.svg` 是三个参数最终最佳值关于种群规模的折线图，
`final_results.csv` 与 `summary.json` 保存数值和配置，另保存各组历史记录用于追溯。
新版运行还会输出 `population_<N>_all_candidates.csv`，记录第 0 代至最后一代全部候选。
此前仅保存最佳解的历史文件无法直接恢复其余候选，需要重新训练才能生成完整记录。

横轴 3、4、5、6 bit 定义为 `log2(population_size)`，不是实际分组线路宽度。
实际局部/全局线路宽度依次为 2/2、2/3、3/4、4/5；总种群 8 时，局部 2 个值补零到
4 个振幅，以保留双量子比特交叉。每组仅一个随机种子，固定代数下种群越大评价次数越多，
因此这项实验不是仅改变线路宽度的控制实验，也不估计重复运行的不确定性。

## 不同实际 qubit 数的噪声验证

运行 `conda run --no-capture-output -n qml python run_noise_validation.py`。
结果保存在 `output/qubit_noise_validation/`。

验证线路由现有 `_circuit()` 生成，再将振幅初始化、交叉和变异全部分解到 `u`、`cx` 基础门。
采用全连接布局，不引入特定设备的连通性或路由。复位理想；不模拟热弛豫、空闲噪声、串扰或泄漏。
`aeqga_noise.py` 中的 `NoiseValidationAEQGA` 复用原来的训练、解码和记录逻辑，
只更改编译方式和模拟后端；普通 `AEQGA` 的默认编译方式保持原样。

| 配置 | 单比特去极化强度 | 双比特去极化强度 | 每个测量比特的对称翻转概率 |
|---|---|---|---|
| ideal | 0 | 0 | 0 |
| low | 0.0001 | 0.001 | 0.005 |
| medium | 0.001 | 0.01 | 0.02 |
| high | 0.005 | 0.05 | 0.05 |
| medium_gate_only | 0.001 | 0.01 | 0 |
| medium_readout_only | 0 | 0 | 0.02 |

去极化强度是 `E(rho)=(1-p)*rho+p*I/2**n` 中的 `p`，并非平均门不保真度。
这些数值是实验假设，未使用真实设备校准。

固定线路部分包含 4 种总种群、2 个分支、3 个参数、4 种门组合和 3 种输入，共 288 条线路。
每个噪声配置对同一线路重复 5 次，每次 1024 shots。另使用密度矩阵计算精确门噪声概率，
再解析施加读出翻转，区分系统性偏移与有限测量噪声。
先检查编译前后理想概率一致，再统一用编译后的理想概率作为解码参考，
避免极小浮点差异通过最小值映射和下界随机替换造成非零的理想解码偏移。

训练部分使用 ideal、medium、high 三个配置，每种 population 取 seed=7、17、27，
共 36 次独立的 100 代训练；全部使用相同初值、参数范围和完整协方差。
每组保存 `best_history.csv` 和 `all_candidates.csv`，含第 0 代至第 100 代。
这是短程噪声敏感性验证，不是此前 1000 代结果的重现，也不是特定设备的上线合格判定。
均值和标准差表示重复优化结果的波动，不表示宇宙学参数置信区间。

| 输出 | 内容 |
|---|---|
| `noise_vs_actual_qubits.png/.svg` | 概率分布偏移与局部/全局解码偏移随实际线路宽度变化 |
| `gate_vs_readout_noise.png/.svg` | 单独门噪声、单独读出噪声和组合噪声的精确对比 |
| `compiled_circuit_resources.png/.svg` | 基础门分解后的 CX 数量与线路深度 |
| `noise_training_final_parameters.png/.svg` | 三个参数和相对经典参考解的最终卡方差，均值与样本标准差 |
| `probabilities_and_counts.npz` | 所有固定线路的精确概率与每次测量计数；未使用的填充概率列为 NaN |
| `fixed_circuit_exact.csv` / `fixed_circuit_samples.csv` | 每条固定线路的精确指标与有限 shots 指标 |
| `training_results.csv` / `training_summary.csv` | 36 次训练的最终结果和分组统计 |
| `noise_validation_report.txt` / `summary.json` | 方法、结果、适用范围、检查、版本及源文件哈希 |

实际宽度不同的分组也存在状态制备、门数和候选输入差异，不能把结果全部归因于 qubit 数本身。
运行噪声模块验证：`conda run -n qml python -m unittest -v test_noise_validation`。

## IBM Kingston / Fez / Marrakesh 校准快照验证

运行 `conda run --no-capture-output -n qml python run_ibm_noise_validation.py`。
默认 population=8、16、32、64，每种设备使用 7、17、27 三个种子，分别运行
有噪声和相同编译方式的无噪声对照；共 72 次训练，每次 100 代、1024 shots。
种群比例仍为 1:1:2，三个参数、数据选择和完整协方差保持同一实验设置。

模型读取 `ibm_backend_snapshots/` 中归档的 IBM qiskit-ibm-runtime 0.50.0
公开 JSON，并由官方 Aer `NoiseModel.from_backend` 构造门噪声及 T1/T2 热弛豫。
同时保留校准中的非对称读出概率。快照日期分别为 Kingston 2026-04-15、
Fez 2025-02-26、Marrakesh 2025-02-26；不是实时查询，也不提交硬件任务。
复现只依赖已有 Qiskit/Aer，无需导入 IBM 云服务客户端或提供账户信息。

物理布局固定为 q0-q4 连通链，各电路取其 2–5 比特前缀。振幅制备、交叉和变异
分解到设备原生门，进行拓扑路由与 ALAP 调度，并对显式等待计入退相干。
复位使用校准时长及热弛豫，但无复位分配错误；不模拟串扰、泄漏、漂移或误差缓解。Kingston 快照中 q1 的平均
读出误差约 16.82%，按原始值保留。结果只适用于所用快照与位置，不能据此排名整台设备。

输出目录 `output/ibm_device_noise_validation/` 和同名 ZIP 包含：

- `ibm_noise_summary.png/.svg`：一张六面板结果图，展示固定电路偏移、最终卡方及三个参数。
- `training_runs/`：每次训练的所有代、全部候选及卡方、最优轨迹和原始量子采样计数。
- `training_results.csv` / `paired_noise_effect.csv`：最终结果及对应无噪声对照的差值。
- `validation_report.txt` / `README_中文.txt`：方法、快照来源、数值结果、检查和适用范围。
- `reproduction/`：复现代码、校准 JSON、输入数据、依赖版本和运行说明。
- `SHA256SUMS.txt`：压缩包内文件校验清单。

每次训练保存第 0 代至第 100 代。标准差描述三个随机种子的优化波动，不是宇宙学参数
置信区间；100 代后的最优值并不保证完全收敛。验证命令：
`conda run -n qml python -m unittest -v test_ibm_noise_validation`。

## 同一代的电路并行合并

`parallel_circuits=K` 把同一代最多 K 条小电路放在互不重叠的 qubit 区域，
构成一条组合电路。仍然只训练一个种群，精英/局部/全局比例和参数解码公式保持原样。
组合电路中每个小电路均获得 `shots` 次测量。根据各自的 classical bit 位置从联合计数中
求边缘计数，再分别解码，因此不会混合三个参数、局部/全局分支的测量结果。

```python
model = AEQGA(function, initial_p, bounds=bounds,
              population_size=64, shots=1024,
              parallel_circuits=6, max_parallel_qubits=150, seed=7)
model.train(x, y, generations=100)
print(model.quantum_circuits_)    # 3 个参数时，100 代共 600 条逻辑小电路
print(model.executed_circuits_)   # 每代全部合并，共执行 100 条组合电路
```

命令行示例（已有 qml 环境）：

```powershell
conda run --no-capture-output -n qml python run_pantheon_aeqga.py --population 64 --generations 100 --shots 1024 --parallel-circuits 6 --max-parallel-qubits 150 --output output/pantheon_parallel
```

对于当前三个参数和 1:1:2 分组，每代只有 6 条逻辑小电路。K 大于 6 时仍只合并现有 6 条；
不会添加训练副本、重复电路或填充活跃 qubit。population=64 的小电路宽度依次是
4、5、4、5、4、5，所以有：

| `parallel_circuits` | 每代实际执行的组合电路数 | 单条组合电路最多活跃 qubit |
|---|---|---|
| 1 | 6 | 5 |
| 2 | 3 | 9 |
| 3 | 2 | 14 |
| 4 | 2 | 18 |
| 6 或更大 | 1 | 27 |

`max_parallel_qubits` 是上限，而非目标 qubit 数。设置为 150 不会自动得到 150 个活跃 qubit。
需要拆分时按原电路顺序形成多个组合电路，它们仍在该代完成后才进入下一代。
默认值 1 保持旧版单电路采样路径；启用合并后使用 MPS 模拟器，利用小电路之间无纠缠的结构，
避免创建长度为 `2**总宽度` 的稠密状态向量。计数拆分也只遍历实际观测到的比特串。
有限 shots 的随机抽样会随合并方式改变，因此不同 K 的相同 seed 不保证相同训练轨迹；
同一环境、相同设置和 seed 可复现。

加入 IBM 公开校准快照及其拓扑：

```powershell
conda run --no-capture-output -n qml python run_pantheon_aeqga.py --device marrakesh --noise-mode noisy --population 64 --generations 100 --shots 1024 --parallel-circuits 6 --max-parallel-qubits 150 --output output/marrakesh_parallel
```

`--device` 可取 `kingston`、`fez`、`marrakesh`；`--noise-mode ideal` 保留相同设备拓扑但关闭噪声。
这些命令在本地模拟，当前入口不提交 IBM 真机任务。Python 可直接使用
`IBMNoiseAEQGA(..., device="marrakesh", parallel_circuits=6)`。

并行 IBM 模式先分配互不重叠的连通区域，再在每个区域内用 SABRE 和 optimization level 3
独立编译，最后按原物理 ID 合并。目标允许的每个原生门均经过检查，路由不会跨越其他分组。
区域分配按拓扑进行确定性的贪心选择，不是全局最优布局，也不保证区域内零 SWAP。
容量不足或区域碎片化时自动拆组；单个电路无法放入有效连通区域则报错。
不完整/失效校准的 qubit 不参加分配，例如所归档 Kingston 快照的 q146 缺少 T1/T2。

IBM 导出的 QPY 可以有 156 条物理线路，其中未使用的线路是空闲占位。
应查看 `active_qubits` 和 `blocks[].qubits` 判断实际活跃数量。
`last_generation_layout.json` 保存这些位置及测量映射；`summary.json` 保存设备快照日期、
哈希和排除的 qubit。旧版单路、2–5 qubit 验证仍使用原 q0–q4 前缀布局。
因此并行与旧版噪声结果的差异还包含物理位置、编译和抽样方式变化。
噪声模型仍不包含串扰、泄漏和漂移；本地验证不能确认真机并行时的吞吐或精度收益。

验证命令：`conda run -n qml python -m unittest -v test_parallel_circuits`。
