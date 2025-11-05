# app.py — Gas Sales Analytics (Landing + Aggregations + Industrial Focus)
# 탭3: [산업용 집중분석] 업종×기간 히트맵 → 셀 클릭: 고객 Top-N / YoY / 다운로드

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 클릭이벤트 모듈: 없으면 화면만 뜨게 우회
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


# ───────────────────────── 유틸 ─────────────────────────
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
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

def yoy_compare(df, key_cols, value_col, period_col, gran: str):
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
    out = pd.merge(a, b, how="left",
                   left_on=key_cols + [period_col],
                   right_on=key_cols + ["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs() > 1e-9,
                         out["증감"] / out["전년동기"] * 100, np.nan)
    return out

@st.cache_data(show_spinner=False)
def read_parquet_any(file):
    return pd.read_parquet(file)  # pyarrow 엔진 자동 사용 (requirements에 명시)

# ───────────────────────── 입력 ─────────────────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종) — Parquet 권장")

# A: 월별 총괄
up_overall = st.sidebar.file_uploader("A) 월별 총괄 (Parquet)", type=["parquet"])
if not up_overall:
    st.info("A(월별 총괄) Parquet 파일을 업로드해줘.")
    st.stop()

overall_raw = read_parquet_any(up_overall)
used_overall = up_overall.name

colsA = overall_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")

def _pickA(keys, default_idx=0):
    for k in keys:
        for c in colsA:
            if k in c:
                return c
    return colsA[default_idx]

c_date = st.sidebar.selectbox("날짜 컬럼", colsA,
    index=colsA.index(_pickA(["날짜","Date","월"])) if _pickA(["날짜","Date","월"]) in colsA else 0)
c_cook = st.sidebar.selectbox("취사용 컬럼", colsA,
    index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 0)

overall = overall_raw.copy()
overall["날짜"] = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"] = overall[c_cook].apply(to_num)

# B: 산업용 상세
up_indetail = st.sidebar.file_uploader("B) 산업용 상세 (여러 파일 업로드 가능, Parquet)", type=["parquet"], accept_multiple_files=True)
if not up_indetail:
    st.info("B(산업용 상세) Parquet 파일을 업로드해줘.")
    st.stop()

used_inds = [f.name for f in up_indetail]
frames = []
for f in up_indetail:
    df = read_parquet_any(f)
    frames.append(df)
indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── B 컬럼 매핑(파일별 컬럼명이 다를 수 있으므로 UI 제공)
colsB = indetail_raw.columns.astype(str).tolist()
st.sidebar.header("③ B(산업용 상세) 컬럼 매핑")

def _pickB(keys, default=None):
    for k in keys:
        for c in colsB:
            if k in c:
                return c
    if default: return default
    return (colsB[0] if colsB else None)

b_date = st.sidebar.selectbox("날짜(월) 컬럼", colsB, index=colsB.index(_pickB(["날짜","청구년월","사용월","년월"])) if _pickB(["날짜","청구년월","사용월","년월"]) in colsB else 0)
b_ind  = st.sidebar.selectbox("업종 컬럼",   colsB, index=colsB.index(_pickB(["업종"])) if _pickB(["업종"]) in colsB else 0)
b_cus  = st.sidebar.selectbox("고객명 컬럼", colsB, index=colsB.index(_pickB(["고객","고객명","거래처"])) if _pickB(["고객","고객명","거래처"]) in colsB else 0)
b_amt  = st.sidebar.selectbox("사용량 컬럼", colsB, index=colsB.index(_pickB(["사용량","Nm3","NM3","m3","수량","MJ"])) if _pickB(["사용량","Nm3","NM3","m3","수량","MJ"]) in colsB else 0)
b_use  = st.sidebar.selectbox("용도 컬럼(선택)", ["<없음>"] + colsB, index=0)

def parse_month_like(s):
    s = str(s)
    for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt).replace(day=1)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

B0 = indetail_raw.copy()
B0["날짜"] = pd.to_datetime(B0[b_date].apply(parse_month_like), errors="coerce")
B0["업종"] = B0[b_ind].astype(str).str.strip()
B0["고객명"] = B0[b_cus].astype(str).str.strip()
B0["사용량"] = pd.to_numeric(B0[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
if b_use != "<없음>":
    B0["용도"] = B0[b_use].astype(str).str.strip()
else:
    B0["용도"] = ""

# ───────────────────────── 전역 범위 ─────────────────────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")

date_min = min(overall["날짜"].min(), B0["날짜"].min())
date_max = max(overall["날짜"].max(), B0["날짜"].max())
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ───────────────────────── 탭 ─────────────────────────
tab0, tab1, tab2 = st.tabs(["🏠 대시보드", "📚 집계", "🏭 산업용 집중분석"])

# ── 탭0
with tab0:
    st.subheader("연도별 용도 누적 스택")
    landing = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    landing["연도"] = landing["날짜"].dt.year
    usage_cols = ["취사용"]
    annual = landing.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")
    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(barmode="stack", template="simple_white", height=420,
                       font=dict(family=FONT, size=13))
    st.plotly_chart(fig0, use_container_width=True)
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1
with tab1:
    st.subheader("집계 — 월/분기/반기/연간")
    gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
    A = overall[(overall["날짜"] >= pd.to_datetime(d1)) & (overall["날짜"] <= pd.to_datetime(d2))].copy()
    A["Period"] = as_period_key(A["날짜"], gran)
    sum_tbl = A.groupby("Period", as_index=False)[usage_cols].sum().sort_values("Period")
    st.dataframe(sum_tbl, use_container_width=True)

# ── 탭2
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    B = B0[(B0["날짜"] >= pd.to_datetime(d1)) & (B0["날짜"] <= pd.to_datetime(d2))].copy()
    # 산업용만 필요하면 필터 (용도 컬럼 있는 경우)
    if "용도" in B.columns and B["용도"].str.contains("산업", na=False).any():
        B = B[B["용도"].str.contains("산업", na=False)]

    if len(B) == 0:
        st.info("선택한 기간/필터에 산업용 데이터가 없습니다.")
        st.stop()

    gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
    B["Period"] = as_period_key(B["날짜"], gran_focus)

    # ① 업종×기간 히트맵
    pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
    pivot = pivot[pivot.columns.sort_values()].sort_index()
    Z = pivot.values
    X = pivot.columns.tolist()
    Y = pivot.index.tolist()

    heat = go.Figure(data=go.Heatmap(
        z=Z, x=X, y=Y, colorscale="Blues", colorbar=dict(title="사용량"),
        hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
    ))
    heat.update_layout(template="simple_white", height=560,
                       xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                       font=dict(family=FONT, size=13),
                       margin=dict(l=70, r=20, t=40, b=40))
    click = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                          override_height=560, override_width="100%")

    # ② 클릭 시: 고객 Top-N & YoY
    if click:
        sel_period = str(click[0].get("x"))
        sel_ind    = str(click[0].get("y"))
        st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")

        yo = yoy_compare(B[B["업종"] == sel_ind], ["업종","고객명"], "사용량", "Period", gran_focus)
        view = yo[yo["Period"] == sel_period].copy().sort_values("사용량", ascending=False)

        view["사용량"]   = view["사용량"].round(0)
        view["전년동기"] = view["전년동기"].round(0)
        view["증감"]     = view["증감"].round(0)
        view["YoY(%)"]  = view["YoY(%)"].round(1)

        top_n = st.slider("상위 N", 5, 100, 20, step=5)
        top_tbl = view.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

        c1, c2 = st.columns([1.3, 1.7])
        with c1:
            st.dataframe(top_tbl.style.format({
                "사용량":"{:,.0f}","전년동기":"{:,.0f}","증감":"{:+,.0f}","YoY(%)":"{:+,.1f}"
            }), use_container_width=True, height=520)
            st.download_button(
                "⬇️ 고객리스트 CSV",
                data=top_tbl.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sel_ind}_{sel_period}_top{top_n}.csv",
                mime="text/csv"
            )
        with c2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_tbl["고객명"], y=top_tbl["사용량"], name="사용량",
                text=[f"{v:,.0f}" for v in top_tbl["사용량"]], textposition="auto"
            ))
            fig_bar.update_layout(template="simple_white", height=520,
                                  xaxis=dict(title="고객명", tickangle=-45),
                                  yaxis=dict(title="사용량"),
                                  font=dict(family=FONT, size=12),
                                  margin=dict(l=40, r=20, t=10, b=120))
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displaylogo": False})
    else:
        st.info("히트맵 셀을 클릭하면 고객 Top-N과 막대그래프가 표시됩니다.")

# ───────────────────────── 파일 정보 ─────────────────────────
with st.expander("🔎 분석에 사용된 원천 파일"):
    st.write(f"A(월별 총괄): **{used_overall}**")
    st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds) > 10 else ""))
