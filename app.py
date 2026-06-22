from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="企业经营数据分析 Copilot",
    page_icon="📊",
    layout="wide",
)


def load_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="gbk")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)

    raise ValueError("不支持的文件格式，请上传 CSV、xlsx 或 xls 文件。")


def build_field_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "字段名": df.columns,
            "字段类型": [str(dtype) for dtype in df.dtypes],
            "非空数量": df.notna().sum().values,
            "缺失值数量": df.isna().sum().values,
            "唯一值数量": df.nunique(dropna=True).values,
        }
    )


def build_quality_warning_list(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []

    empty_name_columns = [
        str(column)
        for column in df.columns
        if str(column).strip() == "" or str(column).startswith("Unnamed:")
    ]
    if empty_name_columns:
        warnings.append(f"发现空列名或未命名列：{', '.join(empty_name_columns)}")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        warnings.append(f"发现重复列名：{', '.join(map(str, duplicate_columns))}")

    full_empty_columns = df.columns[df.isna().all()].tolist()
    if full_empty_columns:
        warnings.append(f"发现全空列：{', '.join(map(str, full_empty_columns))}")

    return warnings


def build_dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["字段名", "缺失率(%)"])

    missing_rate = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > 0].head(8).reset_index()
    missing_rate.columns = ["字段名", "缺失率(%)"]
    if not missing_rate.empty:
        missing_rate["缺失率(%)"] = missing_rate["缺失率(%)"].round(2)
    return missing_rate


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().transpose().round(2)


def build_analysis_input(df: pd.DataFrame) -> str:
    summary = build_dataset_summary(df)
    field_overview_csv = build_field_overview(df).to_csv(index=False)
    missing_summary = build_missing_summary(df)
    warnings = build_quality_warning_list(df)

    parts = [
        "你是一名务实的企业经营数据分析助手。",
        "请基于下面的摘要预处理结果，输出结构化业务分析。",
        "要求输出中文，保持简洁，分为：问题判断、可能原因、建议方向、行动建议。",
        "",
        f"数据行数：{summary['rows']}",
        f"数据列数：{summary['columns']}",
        f"总缺失值：{summary['missing_total']}",
        f"重复行数量：{summary['duplicate_rows']}",
        "",
        "字段概览：",
        field_overview_csv,
    ]

    if not missing_summary.empty:
        parts.extend(["", "缺失率较高字段：", missing_summary.to_csv(index=False)])

    if warnings:
        parts.extend(["", "数据质量提示："] + warnings)

    return "\n".join(parts)


def create_openai_client() -> OpenAI:
    api_key = st.session_state.get("api_key", "").strip()
    base_url = st.session_state.get("base_url", "").strip()

    if not api_key:
        raise ValueError("请先在侧边栏填写 API Key。")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def test_connection(api_key: str, base_url: str, model_name: str) -> tuple[bool, str]:
    if not api_key.strip():
        return False, "请先填写 API Key。"

    try:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip() or None)
        response = client.responses.create(
            model=model_name.strip() or "gpt-4o-mini",
            input="请回复：连接成功。",
            max_output_tokens=32,
        )
        text = response.output_text.strip()
        if text:
            return True, f"连接成功：{text}"
        return True, "连接成功。"
    except Exception as exc:
        return False, f"连接失败：{exc}"


def generate_ai_analysis(df: pd.DataFrame) -> dict:
    client = create_openai_client()
    model_name = st.session_state.get("model_name", "").strip() or "gpt-4o-mini"
    analysis_input = build_analysis_input(df)

    prompt = f"""
请基于以下经营数据摘要，输出严格 JSON，不要输出额外文本。
JSON 格式如下：
{{
  "analysis": {{
    "问题判断": "一句话总结当前经营问题",
    "可能原因": "一句话总结可能原因",
    "建议方向": "一句话总结建议方向"
  }},
  "actions": ["建议1", "建议2", "建议3"]
}}

要求：
1. 使用中文。
2. 只返回 JSON。
3. actions 需要 3 到 5 条，具体、可执行、和数据问题有关。
4. 不要编造数据之外的细节。

经营数据摘要：
{analysis_input}
"""

    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.2,
    )
    return json.loads(response.output_text)


def format_ai_error(exc: Exception) -> str:
    message = str(exc)
    lower_message = message.lower()
    if "api key" in lower_message or "401" in lower_message or "invalid_api_key" in lower_message:
        return "模型调用失败：请检查 API Key、Base URL 和模型名是否正确。"
    return f"模型调用失败：{message}"


def export_report(df: pd.DataFrame, analysis_result: dict | None) -> str:
    summary = build_dataset_summary(df)
    field_overview = build_field_overview(df).to_markdown(index=False)
    missing_summary = build_missing_summary(df)
    numeric_summary = build_numeric_summary(df)

    lines = [
        "# 企业经营数据分析报告",
        "",
        "## 数据概览",
        f"- 行数：{summary['rows']}",
        f"- 列数：{summary['columns']}",
        f"- 总缺失值：{summary['missing_total']}",
        f"- 重复行数量：{summary['duplicate_rows']}",
        "",
        "## AI 分析前摘要预处理",
        field_overview,
    ]

    if not missing_summary.empty:
        lines.extend(["", "### 缺失率较高字段", missing_summary.to_markdown(index=False)])

    if not numeric_summary.empty:
        lines.extend(["", "### 数值字段统计", numeric_summary.to_markdown()])

    if analysis_result:
        analysis = analysis_result.get("analysis", {})
        actions = analysis_result.get("actions", [])
        lines.extend(
            [
                "",
                "## AI 分析结论",
                f"- 问题判断：{analysis.get('问题判断', '')}",
                f"- 可能原因：{analysis.get('可能原因', '')}",
                f"- 建议方向：{analysis.get('建议方向', '')}",
                "",
                "## 行动建议",
            ]
        )
        lines.extend([f"{index}. {item}" for index, item in enumerate(actions, start=1)])

    return "\n".join(lines)


def show_sidebar() -> None:
    st.sidebar.header("模型配置")
    st.sidebar.caption("支持 OpenAI 兼容接口，也支持第三方中转站。")
    st.sidebar.text_input("API Key", type="password", key="api_key")
    st.sidebar.text_input("Base URL", placeholder="例如：https://api.openai.com/v1", key="base_url")
    st.sidebar.text_input("Model", placeholder="例如：gpt-4o-mini", key="model_name")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("测试连接", use_container_width=True):
        ok, message = test_connection(
            st.session_state.get("api_key", ""),
            st.session_state.get("base_url", ""),
            st.session_state.get("model_name", ""),
        )
        if ok:
            st.sidebar.success(message)
            st.session_state["connection_ok"] = True
        else:
            st.sidebar.error(message)
            st.session_state["connection_ok"] = False

    if col2.button("加载示例", use_container_width=True):
        st.session_state["sample_mode"] = True

    st.sidebar.caption("配置仅保留在当前会话内。")


def get_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "月份": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"],
            "销售额": [120000, 118000, 112000, 98000, 103000, 126000],
            "成本": [70000, 69000, 68000, 66000, 67000, 71000],
            "订单数": [320, 315, 300, 280, 290, 335],
            "客户数": [210, 205, 198, 190, 192, 220],
            "渠道": ["线上", "线上", "线下", "线上", "线下", "线上"],
        }
    )


def show_preprocess_area(df: pd.DataFrame) -> None:
    st.subheader("AI 分析前的摘要预处理")
    st.caption("这里展示的是发给模型之前的结构化摘要，方便你解释分析输入从哪里来。")

    summary = build_dataset_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", summary["rows"])
    c2.metric("列数", summary["columns"])
    c3.metric("总缺失值", summary["missing_total"])
    c4.metric("重复行数量", summary["duplicate_rows"])

    st.markdown("**字段概览**")
    st.dataframe(build_field_overview(df), use_container_width=True)

    missing_summary = build_missing_summary(df)
    st.markdown("**缺失率较高字段**")
    if missing_summary.empty:
        st.info("当前没有明显缺失字段。")
    else:
        st.dataframe(missing_summary, use_container_width=True)

    warnings = build_quality_warning_list(df)
    if warnings:
        for item in warnings:
            st.warning(item)
    else:
        st.success("未发现空列名、重复列名或全空列。")

    numeric_summary = build_numeric_summary(df)
    st.markdown("**数值字段统计**")
    if numeric_summary.empty:
        st.info("当前数据中没有数值字段。")
    else:
        st.dataframe(numeric_summary, use_container_width=True)

    st.markdown("**用于模型的摘要文本**")
    st.code(build_analysis_input(df), language="text")


def show_charts(df: pd.DataFrame) -> None:
    st.subheader("基础图表")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    category_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    left, right = st.columns(2)

    with left:
        st.markdown("**数值字段分布**")
        if numeric_columns:
            numeric_col = st.selectbox("选择数值字段", numeric_columns, key="numeric_chart")
            numeric_data = df[numeric_col].dropna()
            if numeric_data.empty:
                st.info("该字段没有可用数值。")
            else:
                chart = (
                    alt.Chart(pd.DataFrame({numeric_col: numeric_data}))
                    .mark_bar()
                    .encode(
                        alt.X(f"{numeric_col}:Q", bin=True, title=numeric_col),
                        alt.Y("count()", title="数量"),
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("没有可用于分布图的数值字段。")

    with right:
        st.markdown("**类别字段 Top 10**")
        if category_columns:
            category_col = st.selectbox("选择类别字段", category_columns, key="category_chart")
            top_n = df[category_col].dropna().astype(str).value_counts().head(10).reset_index()
            top_n.columns = [category_col, "数量"]
            if top_n.empty:
                st.info("该字段没有可用类别值。")
            else:
                chart = (
                    alt.Chart(top_n)
                    .mark_bar()
                    .encode(
                        alt.X("数量:Q", title="数量"),
                        alt.Y(f"{category_col}:N", sort="-x", title=category_col),
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("没有可用于 Top N 的类别字段。")


def render_analysis_panel(df: pd.DataFrame) -> None:
    st.subheader("AI 分析结果")
    st.caption("先看摘要预处理，再手动点击开始分析。")

    if "analysis_result" in st.session_state and st.session_state["analysis_result"]:
        result = st.session_state["analysis_result"]
        analysis = result.get("analysis", {})
        actions = result.get("actions", [])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**问题判断**")
            st.write(analysis.get("问题判断", "暂无结果"))
            st.markdown("**可能原因**")
            st.write(analysis.get("可能原因", "暂无结果"))
        with col2:
            st.markdown("**建议方向**")
            st.write(analysis.get("建议方向", "暂无结果"))
            st.markdown("**行动建议**")
            if actions:
                for index, item in enumerate(actions, start=1):
                    st.write(f"{index}. {item}")
            else:
                st.info("暂无行动建议。")

    if st.button("开始分析", type="primary"):
        if not st.session_state.get("api_key", "").strip():
            st.error("请先在侧边栏填写 API Key。")
            return
        try:
            with st.spinner("正在生成分析结果..."):
                result = generate_ai_analysis(df)
            st.session_state["analysis_result"] = result
            st.success("分析完成。")
            st.rerun()
        except Exception as exc:
            st.error(format_ai_error(exc))

    if st.session_state.get("analysis_result"):
        report = export_report(df, st.session_state["analysis_result"])
        st.download_button(
            "导出 Markdown 报告",
            data=report,
            file_name="enterprise_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


def main() -> None:
    show_sidebar()

    st.title("企业经营数据分析 Copilot")
    st.write(
        "这是一个可直接在浏览器中使用的企业经营数据分析网页，支持上传经营汇总表、"
        "测试第三方中转站连接、手动触发分析并导出报告。"
    )

    if st.session_state.pop("sample_mode", False):
        st.session_state["uploaded_df"] = get_sample_df()
        st.session_state["analysis_result"] = None
        st.success("已加载经营汇总示例数据。")

    uploaded_file = st.file_uploader("上传经营汇总表", type=["csv", "xlsx", "xls"])

    current_df: pd.DataFrame | None = None
    if uploaded_file is not None:
        try:
            current_df = load_file(uploaded_file)
            st.session_state["uploaded_df"] = current_df
            st.session_state["analysis_result"] = None
            st.success("文件读取成功。")
        except Exception as exc:
            st.error(f"文件读取失败：{exc}")

    if current_df is None:
        current_df = st.session_state.get("uploaded_df")

    if current_df is None:
        st.info("请先上传文件，或点击侧边栏的“加载示例”。")
        return

    if current_df.empty:
        st.warning("当前数据为空，请上传有效经营汇总表。")
        return

    st.markdown("---")
    st.subheader("数据预览")
    st.dataframe(current_df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    summary = build_dataset_summary(current_df)
    c1.metric("行数", summary["rows"])
    c2.metric("列数", summary["columns"])
    c3.metric("总缺失值", summary["missing_total"])
    c4.metric("重复行数量", summary["duplicate_rows"])

    st.markdown("---")
    show_preprocess_area(current_df)

    st.markdown("---")
    show_charts(current_df)

    st.markdown("---")
    render_analysis_panel(current_df)


if __name__ == "__main__":
    main()
