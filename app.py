# app.py — Marketing Insight (가볍게 뜨는 기본판 / 클릭 기능은 선택적)
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 클릭 기능 모듈: 있으면 사용, 없으면 비활성
HAS_PLOTLY_EVENTS = False
try:
    from streamlit_plotly_events import plotly_events  # optional
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ───────────────────────── Utils
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    d = pd.to_datetime(dt)
    if gran == "월":
        return d.dt.to_period("M").astype(str)
    elif gran == "분기":
        return d.dt.to_period("Q").astype(str)
    elif gran == "반기":
        y = d.dt.year.astype(str)
        h = np.where(d.dt.month <= 6, "H1", "H2")
        return y + h
    else:
        return d.dt.year.astype(str)

@st.cache_data(show_spinner=False)
def read_parquet(buf):
    return pd.read_parquet(buf)

def ensure_cols(df, need):
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.warning(f"필수 컬럼이 없습니다: {missing}")
        return False
    return True

# ───────────────────────── Sidebar: 업로드
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산) Parquet · B: 산업용 상세(고객/업종) Parquet")

up_overall = st.sidebar.file_uploader("A) 월별 총괄(.parquet)", type=["parquet"])
up_indetail = st.sidebar.file_uploader("B) 산업용 상세(.parquet)", type=["parquet"], accept_multiple_files=True)

if not up_overall:
    st.info("A(월별 총괄) Parquet 파일을 업로드해줘.")
    st.stop()

overall_raw = read_parquet(up_overall)

# 최소 매핑(필수 두 가지만 먼저)
colsA = list(map(str, overall_raw.columns))
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")
c_date = st.sidebar.selectbox("날짜 컬럼", colsA, index=0)
c_cook = st.sidebar.selectbox("취사용(or 대표지표) 컬럼", colsA, index=min(1, len(colsA)-1))

overall = overall_raw.copy()
overall["날짜"] = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"] = to_num(overall[c_cook])

# B 업로드가 없으면 히트맵 탭은 비활성
if up_indetail:
    frames = []
    for f in up_indetail:
        frames.append(read_parquet(f))
    indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
else:
    indetail_raw = pd.DataFrame()

# ───────────────────────── 기간
st.title("📊 도시가스 판매량 분석")
date_min = overall["날짜"].min()
date_max = overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ───────────────────────── Tabs
tabs = ["🏠 대시보드", "📚 집계"]
if not indetail_raw.empty:
    tabs.append("🏭 산업용 집중분석")
tab0, tab1, *rest = st.tabs(tabs)
tab2 = rest[0] if rest else None

# ── 탭0: 대시보드(연도 스택)
with tab0:
    st.subheader("연도별 취사용 누적 스택")
    A = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    A["연도"] = A["날짜"].dt.year
    annual = A.groupby("연도", as_index=False)[["취사용"]].sum().sort_values("연도")

    fig0 = go.Figure()
    fig0.add_trace(go.Bar(x=annual["연도"], y=annual["취사용"], name="취사용"))
    fig0.update_layout(
        barmode="stack", template="simple_white", height=420,
        xaxis=dict(title="Year"), yaxis=dict(title="사용량"),
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    st.plotly_chart(fig0, use_container_width=True)
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1: 집계(월/분기/반기/연간)
with tab1:
    st.subheader("집계 — 월/분기/반기/연간")
    gran = st.radio("집계 단위", ["월", "분기", "반기", "연간"], horizontal=True, key="granularity")
    A = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    A["Period"] = as_period_key(A["날짜"], gran)
    sum_tbl = A.groupby("Period", as_index=False)[["취사용"]].sum().sort_values("Period")

    c1, c2 = st.columns([2,3])
    with c1:
        st.dataframe(sum_tbl.style.format({"취사용": "{:,.0f}"}), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["취사용"], name="취사용"))
        fig.update_layout(template="simple_white", height=360,
                          xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
                          font=dict(family=FONT, size=13))
        st.plotly_chart(fig, use_container_width=True)

# ── 탭2: 산업용 집중분석(선택적)
if tab2 is not None:
    with tab2:
        st.subheader("산업용 집중분석 — 업종×기간 히트맵")
        # 기대 컬럼 존재여부 점검
        need_cols = {"날짜", "업종", "사용량"}
        if not ensure_cols(indetail_raw, need_cols):
            st.stop()

        gran_f = st.radio("기간 단위", ["월", "분기", "반기", "연간"], horizontal=True, key="gran_focus")
        B = indetail_raw.copy()
        B["날짜"] = pd.to_datetime(B["날짜"], errors="coerce")
        B = B[(B["날짜"] >= pd.to_datetime(d1)) & (B["날짜"] <= pd.to_datetime(d2))]
        B["Period"] = as_period_key(B["날짜"], gran_f)

        pvt = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
        pvt = pvt[pvt.columns.sort_values()].sort_index()

        heat = go.Figure(data=go.Heatmap(
            z=pvt.values, x=pvt.columns.tolist(), y=pvt.index.tolist(),
            colorscale="Blues", colorbar=dict(title="사용량"),
            hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(template="simple_white", height=560,
                           xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                           font=dict(family=FONT, size=13), margin=dict(l=70, r=20, t=40, b=40))

        if HAS_PLOTLY_EVENTS:
            clicked = plotly_events(
                heat, click_event=True, hover_event=False, select_event=False,
                override_height=560, override_width="100%"
            )
        else:
            st.plotly_chart(heat, use_container_width=True)
            clicked = []

        if clicked:
            sel_period = str(clicked[0].get("x"))
            sel_ind = str(clicked[0].get("y"))
            st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")
            sub = B[(B["업종"] == sel_ind) & (B["Period"] == sel_period)]
            top = (sub.groupby("고객명", as_index=False)["사용량"]
                      .sum().sort_values("사용량", ascending=False).head(20))
            st.dataframe(top.style.format({"사용량": "{:,.0f}"}), use_container_width=True)

# ───────────────────────── 사용 파일 확인
with st.expander("🔎 분석에 사용된 파일"):
    st.write(f"A 파일: **{getattr(up_overall, 'name', '메모리 업로드')}**")
    if up_indetail:
        st.write("B 파일:", ", ".join([f.name for f in up_indetail]))
    else:
        st.write("B 파일: (없음)")
