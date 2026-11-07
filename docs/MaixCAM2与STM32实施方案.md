# MaixCAM 2 与 STM32 分拣实现

## 系统边界

- MaixCAM 2 固定在整机框架上，以固定斜视角覆盖托盘；云台不参与运动。
- MaixCAM 2 识别形状、颜色和托盘坐标，经 UART 把一个目标发给 STM32。
- STM32 控制 XY 滑台、夹爪、底部 360°舵机转台和显示屏。
- 投放孔下的光电传感器是分拣成功的唯一计数依据。软件先确认光路恢复，再确认稳定遮挡。
- 成功后显示屏更新并重新拍照；超时失败后 MaixCAM 2 重新拍照，优先重抓仍可见的失败类别；若该类别已经不在托盘中，就继续处理当前检测到的物块。

## 每轮动态分盒

每轮恰有六种不同的形状+颜色组合。稳定识别到的首次出现顺序决定盒号：第一个组合对应 1 号盒，依次到第六个组合对应 6 号盒。同一种组合无论有几件都保持同一盒号。映射在本轮开始时建立，轮次重置时清空。

视觉检测适配器必须保持稳定的检测输出顺序，并给跨帧目标提供稳定 `track_id`。否则“首次识别顺序”或多帧投票可能随检测框排序改变。

## 单斜视视觉流程

1. 固定曝光、白平衡、焦点、分辨率和照明。
2. 使用托盘四角点计算图像到托盘毫米坐标的平面单应变换。
3. 每个实例分别预测 9 种形状，并在分割掩膜内估计 10 种颜色。
4. 同一目标连续至少 3 帧类别一致、位置抖动小于配置阈值，才可参与分盒和抓取。
5. 开局确认恰有六种类别后建立盒位映射；类别不足、多于六种或有低置信度时不启动机械动作。
6. 每次成功或失败投放后重新拍摄。失败后优先用新坐标重试原类别；若重拍确认该类别已不在托盘中，则继续处理其他当前可见类别，避免被过期目标卡住。

单应变换只校正托盘平面。物块高于托盘的抓取中心会产生视差，必须用最终相机和实际物块在托盘不同位置做抓取点校准，并记录最大定位误差。示例角点和颜色原型均为模板值，不能直接用于实机。

颜色支持两种路径：实例分割模型额外输出颜色分类，或对掩膜内 RGB 像素取 Lab 中位数并与实测颜色原型比较。当前仓库实现了后一种分类接口；10 个 Lab 原型必须用最终灯光和实物采集后写入配置，距离/间隔不足时会拒识。

## UART v1

串口默认 115200 baud，8-N-1。STM32 完成转台和 XY 回零后发送 `ACK/OK/sequence=0`；MaixCAM 2 收到该就绪 ACK 前不得发送抓取目标。线路电平需按两块板实际 IO 电压确认后再连接，不能把 RS-232 电平接到逻辑 UART。

帧格式（多字节字段小端）：

```text
53 53 | version:u8 | type:u8 | payload_len:u16 | payload | CRC16:u16
```

状态类 payload 固定为 `status:u8 | reserved:u8=0 | operation_sequence:u16`。STM32 HAL 的 `send_status` 回调可使用 `ss_encode_status_frame()` 生成完整应答帧。

CRC 为 CRC-16/CCITT-FALSE，覆盖 `version` 到 payload 末尾。最大 payload 为 128 字节。`VISION_RESULT` payload 固定 18 字节：

```text
shape_id:u8 | color_id:u8 | bin_id:u8 | reserved:u8
tray_x_mm:f32 | tray_y_mm:f32 | confidence:f32 | frame_id:u16
```

消息类型：`VISION_RESULT`、`ACK`、`ACTION_STARTED`、`DROP_SUCCESS`、`DROP_FAILED`、`MOTION_ERROR`、`BIN_POSITION_ERROR`、`EMERGENCY_STOP`、`RESET_ROUND`。UART 接收必须支持任意分片、噪声重同步、长度检查和 CRC 检查；忙碌状态不接收新目标。

## STM32 执行状态

```text
启动 → 转台找索引 → XY回零 → 等待目标 → 移到目标
→ 下降 → 夹紧 → 抬升 → 移到投放孔 → 转到目标盒 → 松爪
→ 等光电成功/超时 → 更新显示或报告失败 → 等待新视觉结果
```

转台保留连续旋转 360°舵机，索引传感器用于建立原点；六个盒位的舵机动作时间须逐一实测。上电先找索引，每轮从原点建立位置基准。光电等待超时则发送 `DROP_FAILED`；此类失败不设重试次数上限。急停、运动超时、限位异常和盒位定位错误进入锁存故障，必须复位后恢复。

光电布防与释放期捕获（O-04）：转台对位阶段持续记录光路状态，只有在对位或释放期间看到光路畅通后才布防，持续遮挡（卡料、上一件残留）不会计为本件。夹爪释放期间也采样光电；去抖确认的落料先保留，等释放动作完成后才报告成功、更新显示并允许下一目标，防止松爪未完成就开始新动作。跨越释放结束时刻的遮挡脉冲连续计时，不会被清零。投放超时仍从释放完成开始计。前提是主循环在对位和释放期间持续调用 tick；短于采样周期的脉冲仍需 O-05 实测后决定是否改为中断捕获。

显示屏只在经过软件去抖的光电成功事件后更新。显示内容由 STM32 根据 shape/color/bin ID 的本地表生成，不采信 MaixCAM 2 的成功计数。

## 当前代码

- `vision/runtime/calibration.py`：四点平面单应变换，不依赖 OpenCV。
- `vision/runtime/assignment.py`：稳定检测、多帧一致性和动态六盒映射。
- `vision/runtime/color.py`：掩膜 RGB 样本的 Lab 原型分类与拒识。
- `vision/runtime/protocol.py`：UART v1 编码、解码和分片流解析。
- `vision/runtime/session.py`：每轮映射、单目标握手、成功计数和失败重拍/重试逻辑。
- `vision/runtime/pipeline.py`：把模型适配器输出的实例、掩膜颜色样本和像素中心转换为稳定检测与 UART 目标。
- `firmware/stm32/`：HAL 无关的协议头和状态机。电机、定时器、索引输入、光电输入、显示屏和 UART DMA 由具体 STM32 工程适配。
- `firmware/stm32/sorting_stream.c/.h`：平台无关的 UART 字节流分帧（U-05）。任意分片、粘包、噪声、错误版本/长度/CRC 后都能重新同步，缓存固定为 136 字节；重同步规则与 Python `StreamDecoder` 一致。板上把中断或 DMA 收到的字节交给 `ss_stream_feed()`，回调中调用 `sort_controller_accept_frame()`；UART 空闲（IDLE）时调用 `ss_stream_idle()`，避免伪帧头卡住后面的真实帧。不要在中断和主循环里同时调用同一个解析器。

桌面验证命令（需要 Python 3 和 GCC）：

```powershell
python -m unittest discover -s vision\tests -v
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware\stm32 firmware\stm32\test_sorting_protocol.c -o $env:TEMP\sorting_protocol_test.exe
& $env:TEMP\sorting_protocol_test.exe
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware\stm32 firmware\stm32\sorting_controller.c firmware\stm32\test_sorting_controller.c -o $env:TEMP\sorting_controller_test.exe
& $env:TEMP\sorting_controller_test.exe
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware\stm32 firmware\stm32\sorting_stream.c firmware\stm32\sorting_controller.c firmware\stm32\test_sorting_stream.c -o $env:TEMP\sorting_stream_test.exe
& $env:TEMP\sorting_stream_test.exe
```

## 配置与验收

- 复制 `vision/runtime/config.example.json` 并按 MaixCAM 2 实际 UART 设备、分辨率和超时配置。
- 将 `vision/configs/tray_calibration.example.json` 中示例像素点替换为实际标定点，用 `python -m vision.runtime.calibrate_tray` 生成运行参数。
- 用最终镜头、灯光和 10 种实物颜色生成 `vision/configs/colors.json`；示例中的原型不可直接部署。
- 完成六类开局映射测试、托盘中心/边缘抓取校准、连续 UART 丢帧/CRC 测试、光电去抖与超时测试、转台六盒索引重复性测试。
- 只有在 MaixCAM 2 上的实际模型、真实相机和 STM32 HAL 适配完成后，才可评估比赛节拍和识别准确率。本仓库目前没有本轮真实数据、训练完成的 9 类模型或具体 STM32 板级工程。
