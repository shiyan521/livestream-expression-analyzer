# 自托管部署指南

## 本地部署

### 前置条件

- Python 3.10+
- ffmpeg（需在系统 PATH 中）

### 部署步骤

1. **Clone 项目**
```bash
git clone https://github.com/your-org/livestream-expression-analyzer.git
cd livestream-expression-analyzer
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt

# 如需 ASR 功能
pip install -r requirements-optional.txt
```

4. **启动服务**
```bash
python web/app.py
```

服务默认运行在 `http://0.0.0.0:5000`。

5. **使用 Gunicorn 生产部署（可选）**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "web.app:app"
```

### 配置环境变量

创建 `.env` 文件或在启动前设置环境变量：

```bash
# 阿里云百炼 API Key（使用百炼后端时需要）
export DASHSCOPE_API_KEY="your-api-key"

# 可选：自定义端口
export FLASK_PORT=5000
```

## Docker 部署

### Dockerfile

在项目根目录创建 `Dockerfile`：

```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt requirements-optional.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt
# 可选依赖按需安装
# RUN pip install --no-cache-dir -r requirements-optional.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "web/app.py"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t livestream-expression-analyzer .

# 运行容器
docker run -d \
    --name expression-analyzer \
    -p 5000:5000 \
    -e DASHSCOPE_API_KEY="your-api-key" \
    livestream-expression-analyzer
```

### Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  expression-analyzer:
    build: .
    container_name: expression-analyzer
    ports:
      - "5000:5000"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
    volumes:
      # 可选：挂载临时文件目录
      - ./tmp:/app/tmp
    restart: unless-stopped
```

启动：

```bash
docker-compose up -d
```

### Docker 部署注意事项

- 基础镜像不包含 Whisper 等可选依赖，如需使用请在 Dockerfile 中取消注释
- Whisper 首次运行会下载模型，建议挂载模型缓存目录：
  ```bash
  docker run -d \
      -v ~/.cache/whisper:/root/.cache/whisper \
      -p 5000:5000 \
      livestream-expression-analyzer
  ```
- 如需 GPU 加速 Whisper，需使用 nvidia-docker 并安装 CUDA 依赖

## GitHub Pages 部署 Demo 网站

如果你只想展示 Demo 报告（不需要实际分析功能），可以将静态文件部署到 GitHub Pages。

### 部署步骤

1. **创建 GitHub Pages 分支**

```bash
git checkout -b gh-pages
```

2. **复制静态文件到根目录**

```bash
# 将 web/static/ 目录内容复制到根目录
cp -r web/static/* .
```

3. **修改报告页数据加载方式**

由于 GitHub Pages 不支持后端 API，需要确保 `report.html` 使用内嵌的 Demo 数据而非 fetch 请求。当前版本的 `report.html` 已经内置了默认 Demo 数据，会自动 fallback 到内嵌数据。

4. **提交并推送**

```bash
git add .
git commit -m "Deploy demo to GitHub Pages"
git push origin gh-pages
```

5. **配置 GitHub Pages**

在 GitHub 仓库设置中：
- 进入 Settings > Pages
- Source 选择 `gh-pages` 分支
- 保存后等待部署完成

6. **访问 Demo**

部署完成后，访问：
```
https://your-username.github.io/livestream-expression-analyzer/
```

### GitHub Pages Demo 限制

- 仅展示 Demo 报告，无法进行实际音频分析
- 历史对比数据为示例数据
- 无法使用 ASR 后端功能

## Nginx 反向代理

如果需要通过域名访问或配置 HTTPS，可以使用 Nginx 反向代理。

### Nginx 配置示例

```nginx
server {
    listen 80;
    server_name expression-analyzer.your-domain.com;

    # 上传文件大小限制（根据需要调整）
    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 支持（如需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # 超时设置（分析可能耗时较长）
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

### 配置 HTTPS

使用 Let's Encrypt 免费证书：

```bash
sudo certbot --nginx -d expression-analyzer.your-domain.com
```

## 配置文件说明

### 环境变量

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key | 无 | 使用百炼后端时必填 |
| `FLASK_PORT` | Web 服务端口 | 5000 | 否 |
| `FLASK_HOST` | Web 服务监听地址 | 0.0.0.0 | 否 |

### 目录结构

```
livestream-expression-analyzer/
├── web/
│   ├── app.py              # Flask 服务入口
│   └── static/
│       ├── index.html      # 首页
│       ├── report.html     # 报告页
│       └── demo-data.json  # Demo 数据
├── tools/
│   ├── __init__.py
│   ├── convert_transcript.py  # 逐字稿转换工具
│   └── merge_asr.py          # ASR 分片合并工具
├── docs/
│   ├── getting-started.md     # 快速开始
│   ├── self-hosting.md        # 自托管部署（本文档）
│   ├── analysis-framework.md  # 分析框架
│   └── images/                # 截图目录
├── requirements.txt           # 基础依赖
├── requirements-optional.txt  # 可选依赖
├── LICENSE
└── README.md
```
