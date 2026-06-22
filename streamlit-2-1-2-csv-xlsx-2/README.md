# 企业经营数据分析 Copilot

这是一个可直接在浏览器中使用的企业经营数据分析网页。

## 当前能力

- 上传 CSV / Excel 经营汇总表
- 读取为 pandas DataFrame
- 展示数据预览和基础指标
- 展示 AI 分析前的摘要预处理
- 支持第三方 OpenAI 兼容中转站
- 手动测试连接和手动开始分析
- 导出 Markdown 报告
- 一键加载示例经营汇总数据

## 本地运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 公网部署

优先使用 Streamlit Cloud。

部署时只需要代码仓库和依赖文件即可。当前版本不要求在云端预置 API Key，用户会在网页侧边栏自行填写：

- API Key
- Base URL
- Model

如果你后续想给自己预置默认值，再单独考虑 Streamlit secrets。

## 使用说明

1. 打开网页。
2. 在侧边栏填写模型配置。
3. 点击“测试连接”。
4. 上传经营汇总表，或者点击“加载示例”。
5. 点击“开始分析”。
6. 下载 Markdown 报告。