# GitHub 上传操作指南

> 本文档供负责上传到 GitHub 的 Agent 或开发者使用。请严格按照以下步骤操作。

## 项目概述

**项目名称**: Livestream Expression Analyzer（直播表达分析器）

**项目描述**: 开源的直播/播客口头表达能力分析系统，20 个维度量化你的表达质量。

**License**: MIT

**技术栈**: Python 3.10+, Flask, librosa, numpy, ffmpeg

## 上传前必做事项

### 1. 替换占位符

项目中以下文件包含 `your-org` 或 `your-username` 占位符，**上传前必须全部替换**为实际的 GitHub 用户名或组织名：

| 文件 | 占位符 | 替换为 |
|------|--------|--------|
| `README.md` | `https://github.com/your-org/livestream-expression-analyzer.git` | 实际仓库地址 |
| `docs/getting-started.md` | `https://github.com/your-org/livestream-expression-analyzer.git` | 实际仓库地址 |
| `docs/self-hosting.md` | `https://github.com/your-org/livestream-expression-analyzer.git` | 实际仓库地址 |
| `docs/self-hosting.md` | `https://your-username.github.io/livestream-expression-analyzer/` | 实际 GitHub Pages 地址 |

**批量替换命令：**
```bash
# 将 YOUR_GITHUB_USERNAME 替换为实际用户名
find . -name "*.md" -exec sed -i 's/your-org/YOUR_GITHUB_USERNAME/g' {} \;
find . -name "*.md" -exec sed -i 's/your-username/YOUR_GITHUB_USERNAME/g' {} \;
```

### 2. 检查个人信息

确认以下内容**不存在**于任何文件中：
- 个人真实姓名
- 个人路径（如 `/Users/xxx/`）
- API Key、Token 等敏感信息
- 私有服务地址

**检查命令：**
```bash
grep -rn "/Users/" . --include="*.py" --include="*.md" --include="*.yaml" --include="*.json"
grep -rn "api_key\|API_KEY\|token\|secret" . --include="*.py" --include="*.yaml"
```

### 3. 删除本指南文件

上传完成后，本文件（`GITHUB_UPLOAD_GUIDE.md`）可以从仓库中删除，或保留作为内部参考。

## 上传步骤

### 方法一：通过 Git 命令行上传

```bash
# 1. 在 GitHub 上创建新仓库
#    仓库名: livestream-expression-analyzer
#    描述: 开源的直播/播客口头表达能力分析系统，20 个维度量化你的表达质量
#    可见性: Public
#    不要勾选 "Initialize this repository"（不要添加 README/LICENSE/.gitignore，项目已包含）

# 2. 进入项目目录
cd livestream-expression-analyzer

# 3. 初始化 Git 仓库
git init
git add .
git commit -m "Initial commit: Livestream Expression Analyzer v1.0

- 20-dimension expression analysis (15 standard + 5 deep)
- Pluggable ASR backends (Whisper/Bailian/Transcript import)
- Web visualization with demo data
- Modular architecture with configuration-driven customization
- MIT License"

# 4. 添加远程仓库并推送
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/livestream-expression-analyzer.git
git branch -M main
git push -u origin main
```

### 方法二：通过 GitHub Web 界面上传

1. 在 GitHub 上创建新仓库（不勾选 Initialize）
2. 将项目目录中所有文件拖拽到 GitHub 上传界面
3. 提交信息填写：`Initial commit: Livestream Expression Analyzer v1.0`

## 上传后配置

### 1. 仓库设置

- **About**: 添加仓库描述和网站链接
- **Topics**: 添加标签 `speech-analysis`, `expression-analysis`, `livestream`, `podcast`, `flask`, `python`, `open-source`
- **License**: GitHub 会自动识别 MIT License

### 2. GitHub Pages（可选，用于 Demo 展示）

如果需要在线展示 Demo 报告：

```bash
# 创建 gh-pages 分支
git checkout -b gh-pages

# 将静态文件复制到根目录
cp -r web/static/* .

# 提交并推送
git add .
git commit -m "Deploy demo to GitHub Pages"
git push origin gh-pages

# 切回 main 分支
git checkout main
```

然后在 GitHub 仓库 Settings > Pages 中：
- Source 选择 `gh-pages` 分支
- 等待部署完成
- 访问 `https://YOUR_GITHUB_USERNAME.github.io/livestream-expression-analyzer/`

### 3. Release 创建（推荐）

创建第一个 Release：

- **Tag**: `v1.0.0`
- **Title**: `Livestream Expression Analyzer v1.0.0`
- **Description**:

```markdown
## 🎉 首个正式版本

### 核心功能
- 20 维度表达分析框架（15 标准层 + 5 深度层）
- 三种 ASR 后端支持（Whisper/百炼/逐字稿导入）
- Web 可视化界面（含 Demo 数据）
- 模块化架构，支持自定义配置

### 文档
- 快速开始指南
- 自托管部署指南（本地/Docker/GitHub Pages）
- 分析框架详解（20 维度）
```

## 项目结构说明

```
livestream-expression-analyzer/
├── README.md                    # 项目主页文档
├── LICENSE                      # MIT 开源协议
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # 基础依赖
├── requirements-optional.txt    # 可选依赖（ASR 等）
├── GITHUB_UPLOAD_GUIDE.md       # 本文件（上传后可删除）
│
├── analyzer/                    # 核心分析模块
│   ├── __init__.py
│   ├── pipeline.py              # 主分析管线
│   ├── acoustic.py              # 声学特征分析
│   ├── text_metrics.py          # 文本指标分析
│   ├── emotion.py               # 情绪与疲劳分析
│   ├── interaction.py           # 互动分析
│   ├── topics.py                # 话题结构分析
│   ├── deep_analysis.py         # 深度分析（5 维度）
│   └── utils.py                 # 工具函数
│
├── asr/                         # ASR 后端模块
│   ├── __init__.py
│   ├── base.py                  # ASR 后端抽象接口
│   ├── factory.py               # 后端工厂
│   ├── whisper_backend.py       # Whisper 本地后端（免费）
│   ├── bailian_backend.py       # 阿里云百炼后端（付费）
│   ├── bl_backend.py            # BL CLI 后端
│   └── transcript_import.py     # 逐字稿导入后端（最准确）
│
├── config/                      # 配置文件
│   ├── fillers.yaml             # 口癖词表（可自定义）
│   ├── speakers.yaml            # 说话人识别配置
│   └── topic_labels.yaml        # 话题标签配置
│
├── diagnosis/                   # 发音诊断模块
│   ├── __init__.py
│   └── pronunciation.py         # 7 维度发音诊断
│
├── docs/                        # 文档
│   ├── getting-started.md       # 快速开始指南
│   ├── self-hosting.md          # 自托管部署指南
│   └── analysis-framework.md    # 分析框架详解
│
├── tools/                       # 工具脚本
│   ├── __init__.py
│   ├── convert_transcript.py    # 逐字稿格式转换
│   ├── merge_asr.py             # ASR 分片合并
│   └── filler_context.py        # 口癖上下文分析
│
└── web/                         # Web 可视化
    ├── app.py                   # Flask 服务
    └── static/
        ├── index.html           # 首页（上传入口）
        ├── report.html          # 报告页（可视化展示）
        └── demo-data.json       # Demo 数据
```

## 验证清单

上传完成后，请逐项验证：

- [ ] GitHub 仓库页面正常显示 README
- [ ] README 中的链接可正常跳转
- [ ] LICENSE 文件被 GitHub 识别为 MIT
- [ ] 仓库 Topics 已添加
- [ ] clone 后 `pip install -r requirements.txt` 可正常安装
- [ ] `python web/app.py` 可正常启动
- [ ] 访问 `http://localhost:5000` 可看到首页
- [ ] Demo 报告页可正常展示
- [ ] 无个人信息泄漏
- [ ] （可选）GitHub Pages Demo 可正常访问
- [ ] （可选）Release v1.0.0 已创建
