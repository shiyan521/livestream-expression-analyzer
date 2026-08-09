# Livestream Expression Analyzer - 静态演示站

本目录是项目的**静态托管版本**（用于魔搭创空间 / 任意静态托管），
包含完整的报告展示前端。无需后端即可在线查看分析报告。

## 文件说明

| 文件 | 说明 |
|------|------|
| `index.html` | 首页（自动检测后端：静态环境进入演示模式，展示真实案例入口） |
| `report.html` | 报告页（支持 `?data=xxx.json` 参数加载指定分析数据） |
| `demo-data.json` | 内置演示数据（0712 单人直播风格） |
| `live-analysis-0805.json` | **真实案例分析**：0805 连麦直播（126 分钟音频 + 224 句逐字稿）全量分析结果 |

## 部署到魔搭创空间（Static）

1. 登录 [modelscope.cn](https://modelscope.cn) → 创空间 → 创建创空间
2. 选择**编程式创建**，接入 SDK 选择 **Static**
3. 将本目录所有文件上传（git push 或网页上传）：
   ```bash
   git clone https://www.modelscope.cn/studios/<你的用户名>/<空间名>.git
   cd <空间名>
   # 复制本目录文件到仓库根目录
   git add . && git commit -m "部署静态演示站" && git push
   ```
4. 构建完成后访问 `https://modelscope.cn/studios/<你的用户名>/<空间名>`
   即可看到首页；点击「0805 连麦直播分析报告」查看真实案例分析。

## 本地运行完整版（含真实分析能力）

```bash
pip install -r requirements.txt
python web/app.py          # 访问 http://localhost:5000（5000 被占用时改 5050）
```

上传音频（MP3/WAV/M4A/AAC/FLAC）+ 逐字稿（腾讯会议/飞书妙记导出、JSON、TXT、DOCX、SRT），
即可执行真实分析：口癖检测、语速句式、声学特征、情绪曲线、话题分段等 20 维度报告。
