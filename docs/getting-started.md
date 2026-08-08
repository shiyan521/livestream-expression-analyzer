# 快速开始指南

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 推荐 3.11 或 3.12 |
| ffmpeg | 任意版本 | 音频处理依赖，需在系统 PATH 中 |
| pip | 最新版 | Python 包管理器 |

### 安装 ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
```bash
# 使用 winget
winget install ffmpeg

# 或从官网下载: https://ffmpeg.org/download.html
```

### 验证安装

```bash
python --version   # 应输出 3.10+
ffmpeg -version    # 应输出 ffmpeg 版本信息
```

## 安装步骤

### 1. Clone 项目

```bash
git clone https://github.com/shiyan521/livestream-expression-analyzer.git
cd livestream-expression-analyzer
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 或使用 conda
conda create -n expression-analyzer python=3.11
conda activate expression-analyzer
```

### 3. 安装基础依赖

```bash
pip install -r requirements.txt
```

基础依赖包含：
- numpy - 数值计算
- librosa - 音频分析
- flask - Web 服务
- ffmpeg-python - ffmpeg 接口
- pyyaml - 配置文件解析

## 三种使用方式

### 方式一：逐字稿导入（推荐，最准确）

如果你已经有会议/直播的逐字稿文件（如腾讯会议导出的逐字稿），这是最推荐的方式。

**适用场景：** 腾讯会议录制、已有字幕文件、人工校对过的逐字稿

**支持格式：** JSON（ASR diarization 格式）、TXT（带说话人标注）、DOCX、SRT

**使用步骤：**

1. 启动 Web 服务
```bash
python web/app.py
```

2. 打开浏览器访问 `http://localhost:5000`

3. 上传音频文件

4. 在"ASR 转写后端"下拉菜单中选择"逐字稿导入（推荐，最准确）"

5. 上传逐字稿文件

6. 点击"开始分析"

**命令行方式转换逐字稿：**

如果你的逐字稿是纯文本格式，可以使用工具脚本转换为标准 JSON 格式：

```bash
# 基本用法
python tools/convert_transcript.py \
    --input transcript.txt \
    --output output.json

# 指定说话人映射
python tools/convert_transcript.py \
    --input transcript.txt \
    --output output.json \
    --speaker-map '{"Speaker A": 0, "Speaker B": 1}'

# 从文件加载说话人映射
python tools/convert_transcript.py \
    --input transcript.txt \
    --output output.json \
    --speaker-map speaker_map.json
```

### 方式二：Whisper 自动转写（免费）

使用 OpenAI 的 Whisper 模型在本地进行语音转写，完全免费。

**适用场景：** 没有逐字稿、希望全自动分析

**安装 Whisper：**

```bash
pip install -r requirements-optional.txt
```

这会安装：
- openai-whisper - 语音识别
- pyannote.audio - 说话人分离
- python-docx - DOCX 文件解析
- playwright - 截图工具

**使用步骤：**

1. 启动 Web 服务
```bash
python web/app.py
```

2. 打开浏览器访问 `http://localhost:5000`

3. 上传音频文件

4. 在"ASR 转写后端"下拉菜单中选择"Whisper 本地转写（免费，需安装）"

5. 点击"开始分析"

**注意事项：**
- 首次使用时会自动下载 Whisper 模型（约 1-3 GB）
- 转写速度取决于硬件，CPU 较慢，GPU 显著加速
- 推荐使用 `medium` 或 `large` 模型以获得更好的中文识别效果

### 方式三：阿里云百炼（付费，需自己申请）

使用阿里云百炼（DashScope）ASR 服务，准确度最高但需要付费。

**适用场景：** 需要高精度转写、长音频处理、预算充足

**配置步骤：**

1. 申请阿里云百炼 API Key
   - 访问 [阿里云百炼控制台](https://dashscope.console.aliyun.com/)
   - 创建 API Key

2. 设置环境变量
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

3. 启动 Web 服务
```bash
python web/app.py
```

4. 打开浏览器访问 `http://localhost:5000`

5. 上传音频文件

6. 在"ASR 转写后端"下拉菜单中选择"阿里云百炼（付费，需配置 API Key）"

7. 点击"开始分析"

## Web 界面使用说明

### 首页

1. 上传音频文件（支持拖拽或点击上传）
2. 选择 ASR 转写后端
3. 如选择"逐字稿导入"，上传逐字稿文件
4. 点击"开始分析"按钮
5. 等待分析完成，自动跳转到报告页

### 报告页

报告页包含以下模块：

- **评分总览** - 综合评分与表达水平
- **综合评分** - 六维度雷达图
- **口癖深度分析** - 核心口癖与延伸词分布
- **语速与句式** - 语速曲线与句长分布
- **声学特征** - 音高、音量、清晰度
- **表达节奏与情绪** - 停顿分布与情绪曲线
- **话题表现** - 按话题拆解的表现对比
- **口癖热点分析** - 时间维度口癖密度热力图
- **历史对比** - 跨场次趋势追踪
- **改善建议** - 分优先级的改善方案
- **改善路径** - 短期/中期/长期改善规划

## 命令行工具

### convert_transcript.py - 逐字稿格式转换

将纯文本逐字稿转换为标准 ASR diarization JSON 格式。

```bash
python tools/convert_transcript.py \
    --input <输入文件> \
    --output <输出文件> \
    [--speaker-map <JSON字符串或文件路径>]
```

### merge_asr.py - 分片 ASR 合并

将分片处理的 ASR JSON 文件合并为完整文件。

```bash
python tools/merge_asr.py \
    --input-dir <分片目录> \
    --output <输出文件> \
    [--offsets <逗号分隔的偏移量列表>]
```

## 常见问题

### Q: 上传后返回了 Demo 数据？

A: 这说明你选择的 ASR 后端未安装或未配置。请根据提示安装对应的后端：
- Whisper: `pip install openai-whisper`
- 百炼: 设置 `DASHSCOPE_API_KEY` 环境变量
- 逐字稿: 确保上传了逐字稿文件

### Q: 分析速度很慢？

A: 分析速度主要取决于 ASR 转写环节：
- 逐字稿导入：最快（跳过转写）
- Whisper 本地：取决于硬件，建议使用 GPU
- 阿里云百炼：取决于网络和服务端处理速度

### Q: 支持哪些语言？

A: 目前主要针对中文表达分析优化。Whisper 后端支持多语言转写，但口癖检测规则以中文为主。
