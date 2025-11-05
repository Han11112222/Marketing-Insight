# app.py — Gas Sales Analytics (Landing + Aggregations + Industrial Focus)
# - 탭3: [산업용 집중분석] 업종×기간 히트맵 → 셀 클릭: 고객 Top-N / YoY / 다운로드

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import time

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# 추가 용도(대시보드 첫 화면에 있으면 함께 스택으로 표시)
CAND_EXTRA = [
    "수송용", "업무용", "연료전지용", "열전용설비용",
    "열병합용", "열병합용1", "열병합용2",
    "일반용", "일반용(1)", "일반용(2)"
]

# ───────────── 공통 유틸 ─────────────
def to_num(x):
    """숫자 변환 함수"""
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    """날짜 데이터를 월/분기/반기/연간으로 변환하는 함수"""
    d = pd.to_datetime(dt)
    if gran == "월":
        return d.dt.to_period("M").astype(str)          # e.g., '2025-09'
    elif gran == "분기":
        return d.dt.to_period("Q").astype(str)          # e.g., '2025Q3'
    elif gran == "반기":
        y = d.dt.year.astype(str)
        h = np.where(d.dt.month <= 6, "H1", "H2")
        return (y + h)                                  # e.g., '2025H1'
    else:
        return d.dt.year.astype(str)                    # e.g., '2025'

def yoy_compare(df, key_cols, value_col, period_col, gran: str):
    """YoY 비교표 생성 함수"""
    lag_map = {"월": 12, "분기": 4, "반기": 2, "연간": 1}
    lag = lag_map.get(gran, 12)
    p = df[period_col].astype(str)

    if gran in ["월", "분기"]:
        prev = (pd.PeriodIndex(p) - lag).astype(str)
    elif gran == "반기":
        y = p.str[:4].astype(int)
        h = p.str[-2:].map({"H1": 1, "H2": 2}).astype(int)
        idx = (y - y.min()) * 2 + (h - 1)
        prev_idx = idx - 2
        base = y.min()
        prev = ((prev_idx // 2) + base).astype(str) + np.where((prev_idx % 2) == 0, "H1", "H2")
    else:
        prev = (p.astype(int) - 1).astype(str)

    cur = df.copy()
    cur["_prev"] = prev
    a = cur.groupby(key_cols + [period_col], as_index=False)[value_col].sum()
    b = (
        cur.rename(columns={period_col: "_prev"})
        .groupby(key_cols + ["_prev"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "전년동기"})
    )
    out = pd.merge(a, b, how="left", left_on=key_cols + [period_col], right_on=key_cols + ["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs() > 1e-9, out["증감"] / out["전년동기"] * 100, np.nan)
    return out

@st.cache_data(show_spinner=False)
def read_parquet_any(path_or_buf):
    """parquet 파일 읽는 함수"""
    try:
        return pd.read_parquet(path_or_buf)
    except Exception as e:
        st.error(f"Error reading parquet file: {e}")
        return None

# ───────────── 데이터 입력 ─────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종)")

up_overall = st.sidebar.file_uploader("A) 월별 총괄 (Parquet)", type=["parquet"])
if up_overall:
    overall_raw = read_parquet_any(up_overall)
    used_overall = up_overall.name
else:
    st.info("A(월별 총괄)를 업로드해주세요.")
    st.stop()

colsA = overall_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")
def _pickA(keys, default_idx=0):
    for k in keys:
        for c in colsA:
            if k in c:
                return c
    return colsA[default_idx]

c_date   = st.sidebar.selectbox("날짜", colsA, index=colsA.index(_pickA(["날짜","Date","월"])) if _pickA(["날짜","Date","월"]) in colsA else 0)
c_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 1)

overall = overall_raw.copy()
overall["날짜"] = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"] = overall[c_cook].apply(to_num)

up_indetail = st.sidebar.file_uploader("B) 산업용 상세 (Parquet)", type=["parquet"], accept_multiple_files=True)
used_inds = []
if up_indetail:
    frames = []
    for f in up_indetail:
        used_inds.append(f.name)
        df = read_parquet_any(f)
        if df is not None:
            frames.append(df)
    indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
else:
    st.info("B(산업용 상세)를 업로드해주세요.")
    st.stop()

# ───────────── 범위 설정 ─────────────
st.title("📊 도시가스 판매량 분석")
date_min = min(overall["날짜"].min(), indetail_raw["날짜"].min()) if len(indetail_raw) > 0 else overall["날짜"].min()
date_max = max(overall["날짜"].max(), indetail_raw["날짜"].max()) if len(indetail_raw) > 0 else overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ───────────── 탭 구성 ─────────────
tab0, tab1, tab2 = st.tabs(["🏠 대시보드", "📚 집계", "🏭 산업용 집중분석"])

# 탭0: 랜딩(연도×용도 스택)
with tab0:
    st.subheader("연도별 용도 누적 스택")
    landing = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    landing["연도"] = landing["날짜"].dt.year
    usage_cols = ["취사용"]  # 필요에 따라 추가 용도들 추가
    annual = landing.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")
    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(barmode="stack", template="simple_white", height=420)
    st.plotly_chart(fig0, use_container_width=True)

# 탭1: 집계
with tab1:
    st.subheader("집계 — 월/분기/반기/연간")
    gran = st.radio("집계 단위", ["월", "분기", "반기", "연간"], horizontal=True, key="granularity")
    A = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    A["Period"] = as_period_key(A["날짜"], gran)
    sum_tbl = A.groupby("Period", as_index=False)[usage_cols].sum().sort_values("Period")
    st.dataframe(sum_tbl)

# 탭2: 산업용 집중분석
with tab2:
    st.subheader("산업용 집중분석 — 업종×기간 히트맵")
    if len(indetail_raw) == 0:
        st.info("산업용 상세 파일(B)이 없어 히트맵을 표시할 수 없습니다.")
        st.stop()

    gran_focus = st.radio("기간 단위", ["월", "분기", "반기", "연간"], horizontal=True, key="gran_focus")
    B = indetail_raw[(indetail_raw["날짜"] >= pd.to_datetime(d1)) & (indetail_raw["날짜"] <= pd.to_datetime(d2))].copy()

    B["Period"] = as_period_key(B["날짜"], gran_focus)
    pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
    Z = pivot.values
    X = pivot.columns.tolist()
    Y = pivot.index.tolist()
    heat = go.Figure(data=go.Heatmap(z=Z, x=X, y=Y, colorscale="Blues", colorbar=dict(title="사용량")))
    st.plotly_chart(heat)

# 사용된 파일 확인
with st.expander("🔎 분석에 사용된 원천 파일"):
    st.write(f"A(월별 총괄): **{used_overall}**")
    st.write(f"B(산업용 상세): {', '.join(used_inds)}")
