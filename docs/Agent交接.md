# 智能分拣项目 Agent 交接

更新日期：2026-11-07。面向另一台 Windows x64 电脑上的 Claude / Opus，首轮目标是理解项目、恢复开发环境、验证测试和演示，然后按任务编号继续开发。

## 接手顺序与依据

1. 先读根目录 [CLAUDE.md](../CLAUDE.md) 和 [AGENTS.md](../AGENTS.md)，检查 `git status --short --branch`，保留已有工作区改动。
2. 读 [README](../README.md)、本文件和 [联合优化模型验收结果](联合优化模型验收结果.md)，了解当前默认版本及其边界。
3. Agent 阅读 [任务跟踪表.md](任务跟踪表.md)，按需查询 CSV 中对应任务的完整字段；用户阅读 [任务跟踪表.xlsx](任务跟踪表.xlsx)。CSV 是维护源，Excel 与 Markdown 由工具同步生成。
4. 再读 [MaixCAM 2 与 STM32 实施方案](MaixCAM2与STM32实施方案.md) 了解硬件分工和接口。[项目总结](项目总结.md)、训练计划和早期推进方案包含历史阶段信息，不能替代当前代码、模型清单和已验证结果。
5. 用户最新决定优先。需求表与方案有冲突时，指出涉及的任务编号和依据；未决事项不得仅凭较新的文件日期视为已确认。

## 当前方案与实现边界

- 视觉设备为 Sipeed MaixCAM 2，固定斜视覆盖托盘，识别形状、颜色和目标坐标，通过双向 UART 与 STM32 通信。
- STM32 控制 XY、Z/夹爪、连续舵机转台和显示屏；转台方案保留舵机并增加索引传感器，具体板卡与现场参数仍待补齐。
- 每轮六种形状与颜色组合按首次稳定识别分配到六个盒，同类进同盒。物块基本散放不重叠，夹爪从上方垂直抓取。
- 投放光电的有效事件用于成功计数，投放超时后重拍重识别并重试；持续低置信度的处理、暂停恢复和空盘判断仍需按任务记录完善。
- 失败物块消失后的处置、同帧并列排序、恢复与结束规则，核对 R-09、R-10、R-11。草案中的建议参数不代表这些任务已经闭环。

Python 的 `vision/runtime/` 已有坐标标定、颜色接口、稳定识别、动态分盒、UART 编解码、pipeline 和 session；`firmware/stm32/` 已有 HAL 无关的协议及控制状态机骨架。桌面测试通过不表示 MaixCAM 2 已部署、STM32 已接线或整机已验收。

## 当前模型与数据

默认入口是根目录 `演示模型.cmd`，加载 `models/trained/stl_shape_joint_v4/pipeline.json`。目录名中的 v4 表示该轮发布，JSON 的运行时格式为 **version 5**，不是版本冲突。

| 入口 | 模型目录 | 用途 |
| --- | --- | --- |
| `演示模型.cmd` | `models/trained/stl_shape_joint_v4/` | 当前联合优化默认流程 |
| `演示首版对照.cmd` | `models/trained/stl_shape_v1/` | 首版对照 |
| `演示专项模型.cmd` | `models/trained/stl_shape_focus_v2/` | 易混形状专项对照 |

第二轮 5800 张新合成场景、11911 个物块的同集对照中，默认流程严格实例得分 **86.01%**，额外误报 **75**，总体边界 IoU **0.8541**。严格得分不是普通分类准确率；总体改善也不代表每一类、每张图都改善。首版及专项版的不同历史测试结果不能直接与本轮得分比较，详见验收报告。

已有 **46 张唯一实拍**参与训练/开发，独立实拍验收仍为 **0**。页面检查曾出现漏检、错类和轮廓碎片；真实场景 99%、板上性能以及完整抓取分拣闭环均未验收。D-08、D-10 至 D-12 及整机任务继续跟踪这些缺口。

导入图片调用本机模型；页面预置样例仍为首版对照。只打开离线 HTML 不会运行新模型，网页上传也不会自动保存图片或加入训练集。

## 仓库带了什么，没带什么

| 内容 | 新机器克隆后是否具备 |
| --- | --- |
| Python/C 代码、配置、测试、任务表、报告 | 是 |
| 三版发布模型与推理所需权重、配置 | 是 |
| `docs/stl_shape_v1_demo/` 的样例与离线页面 | 是，可用于首次上传验证 |
| STL 物块、已提交的托盘模型与机械资料 | 是，最新装配版本仍按 P-05 核实 |
| `dataset/` 的训练图、实拍、标注和冻结数据 | 否，未入库 |
| `runs/segment/` 的完整训练与评估记录 | 否，未入库 |
| `reference/` 的第三方资料与群聊归档 | 否，未入库 |
| `.venv-vision/` 和原机器的服务/账号配置 | 否，新机器重建 |
| RTX5060 双人操作手册 HTML/Markdown | 否，按用户要求保留在原机器本地 |

仓库中的报告可能引用本地未入库的图片或实验记录；首次接手不需要这些材料。需要复现训练或完整历史评估时，再列出具体数据、标注、冻结清单、模型与实验记录的依赖，单独迁移。不要将重复或开发用实拍当作新的独立验收样本。

## 另一台 Windows 的安装步骤

使用 **Windows x64、标准 CPython 3.14（非 free-threaded）和 CPU** 作为首次接手基线。安装 Git、Python 3.14 及承载 Opus 的开发工具，确认 `git --version` 和 `py -3.14 --version` 可用。模型名称以该平台实际提供的选项为准；本仓库不包含 API 密钥或模型服务配置。

在存放项目的父目录打开 PowerShell，逐条执行。若某一步报错，先记录错误，不继续运行后面的步骤。

```powershell
git clone https://github.com/zhangyizhen123/smart-sorting.git
cd smart-sorting
git log -1 --oneline
py -3.14 -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install --upgrade pip
.\.venv-vision\Scripts\python.exe -m pip install -r vision/requirements-handoff.txt
.\.venv-vision\Scripts\python.exe -m pip check
```

若已经克隆过，先检查工作区，再在无本地冲突时使用 `git pull --ff-only`；不要覆盖已有改动或重建一个正在使用的环境。此处直接调用虚拟环境的 Python，无需更改 PowerShell 执行策略或激活脚本。

`vision/requirements-handoff.txt` 锁定首次接手的 CPU 依赖，包含视觉环境和任务表所需的 XlsxWriter/openpyxl。旧 `requirements.txt`、`requirements-focused.txt` 和 `environment.yml` 属于历史训练配置，不与此清单叠加安装；CUDA、付费训练、MaixCAM SDK 和 C 编译器由后续具体任务另行配置。

## 首轮检查与演示

在项目根目录执行：

```powershell
.\.venv-vision\Scripts\python.exe -m unittest discover -s vision/tests -p "test_*.py"
.\.venv-vision\Scripts\python.exe scripts/update_task_tracker.py check
.\演示模型.cmd
```

截至本次交接，现有单元测试基线为 **97 项**；数量随后续开发变化，接手报告应记录实际运行数及失败项。`check` 校验 CSV、Markdown 与 Excel 一致，不会改写表格。

浏览器打开后确认模型就绪，选择 `docs/stl_shape_v1_demo/samples/scene000000__frame000.jpg`，点击“识别形状”，确认能显示预测并切换校正/原始轮廓。保留启动窗口；结束后在该窗口按 Ctrl+C。

需要核对服务时，在第二个 PowerShell 窗口运行：

```powershell
$sortingHealth = Invoke-RestMethod http://127.0.0.1:8765/api/health
$sortingExpectedHash = (Get-FileHash models/trained/stl_shape_joint_v4/pipeline.json -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $sortingHealth.ready -or $sortingHealth.pipeline_sha256 -ne $sortingExpectedHash) { throw '当前服务未加载预期默认模型' }
```

若 8765 已被占用，程序会打印实际端口，请将上面 URL 改为启动窗口中的地址。检查应针对本次启动的服务。需要不自动打开浏览器时，运行 `.\演示模型.cmd --no-open`。

`.gitattributes` 保持已发布模型、配置和算法文件的哈希一致；不要通过禁用校验或重写预期哈希解决版本不匹配。`vision/tests/check_joint_default_launcher.py` 等历史专项检查依赖未入库的评估目录，不能代替上述可在纯克隆仓库中运行的首次检查。

首次只要求 Python 测试、台账一致性和演示。修改 UART 或 STM32 状态机时，再安装支持 C11 的 GCC，在项目根目录运行相应主机测试：

```powershell
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware/stm32 firmware/stm32/test_sorting_protocol.c -o "$env:TEMP/sorting_protocol_test.exe"
& "$env:TEMP/sorting_protocol_test.exe"
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware/stm32 firmware/stm32/sorting_controller.c firmware/stm32/test_sorting_controller.c -o "$env:TEMP/sorting_controller_test.exe"
& "$env:TEMP/sorting_controller_test.exe"
```

## 发给 Opus 的第一条消息

在承载 Opus 的开发工具中打开克隆后的项目目录，再发送：

```text
请接手这个智能分拣项目，先完成项目理解和开发环境自检。

依次阅读 CLAUDE.md、AGENTS.md、README.md、docs/Agent交接.md、
docs/任务跟踪表.md 和 docs/联合优化模型验收结果.md。
开始具体任务前，再读取 CSV 中该编号的完整字段。

项目由 MaixCAM 2 负责视觉、STM32 负责运动控制；目标是九种形状、
十种颜色，每轮六类组合动态分配六个盒。当前默认是联合优化模型，
合成测试结果不能当作真实分拣准确率。训练数据和完整实验目录
没有随 Git 仓库提供。

检查 Git 状态，按交接文档建立 CPU 环境，运行现有单元测试和
任务表检查，启动默认演示并验证仓库样例上传。

首次回复请说明当前架构和默认版本、克隆提交号、实际检查结果、
软件待办与现场硬件缺口，以及建议下一步推进的三个任务编号和理由。
首次检查不启动新训练、不消费云资源、不替换默认模型。

遵守任务表约定，完成并验证任务后立即更新证据和状态；不要推断
线下硬件进度。需求有冲突时列出依据，不自行替用户确定规则。
```

## 后续工作与交接验收

可先评估的三个软件任务：**U-05**（C UART 分片及噪声重同步）、**O-04**（释放期间光电脉冲捕获）、**V-09**（六类建表与未知类别边界）。先核对各行依赖与当前代码，首次接手只给出建议，后续按用户指定的任务推进。I-02 的跨端完整互操作测试依赖 U-05；不要忽略依赖直接标为完成。

需用户/队友提供的资料仍包括最终 CAD 装配与相机位置（P-05）、STM32 型号及接线（S-03/U-04）、最终相机实拍与标定/颜色参数（C/D 系列）、转台/光电/显示屏参数及现场记录（T/O 系列）。独立实拍、板级部署、整机抓取分拣的验收分别记账。

本机已在全新 Windows CPU 环境中安装全部 46 个锁定依赖，使用不含训练数据的 Git 检出通过 97 项单元测试、台账检查，以及默认启动器、模型哈希、页面和样例 API 上传验证；原始与校正轮廓均返回。详见[本机隔离验证记录](../runs/task_audit/2026-11-07_opus_handoff.json)。这不代替另一台机器的实际验收。

**M-07** 记录交接材料和本机干净环境的复现验证；**M-08** 记录另一台 Windows 上的实际接手。目标机器需提供克隆提交号、Python 版本、安装/测试输出、演示结果及 Opus 首轮报告，取得证据后再更新 M-08。完整任务状态始终以任务表为准，不在本文复制整份台账。
