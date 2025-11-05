# app.py — Marketing Insight (Parquet 우선, 업로드/레포 자동탐색, 빠른 실행)
# 탭: 대시보드 / 집계 / 산업용 집중분석(업종×기간 히트맵 → 클릭시 고객 Top-N)

import os, glob, io, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ================= 기본 세팅 =================
st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ----- 유틸
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    if gran == "월":
        return d.dt.to_period("M").astype(str)
    elif gran == "분기":
        return d.dt.to_period("Q").astype(str)
    elif gran == "반기":
        y = d.dt.year.astype(str)
        h = np.where(d.dt.month <= 6, "H1", "H2")
        return (y + h)
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
    out["YoY(%)"] = np.where(out.get("전년동기", 0).abs() > 1e-9,
                         out["증감"] / out["전년동기"] * 100, np.nan)
    return out

@st.cache_data(show_spinner=False)
def read_parquet_any(file_or_bytes):
    return pd.read_parquet(file_or_bytes)

def find_repo_parquets(pattern="*.parquet"):
    files = sorted(glob.glob(pattern))
    # 대용량 피하기 위해 0바이트/깨진 파일 제외
    return [f for f in files if os.path.getsize(f) > 0]

# ================= 사이드바: 데이터 입력 =================
st.sidebar.header("① 데이터 입력 (Parquet 권장)")
st.sidebar.caption("A(월별 총괄) 1개, B(산업용 상세) 1개 이상 가능")

# A 업로드
up_A = st.sidebar.file_uploader("A) 월별 총괄(.parquet)", type=["parquet"])
# B 업로드 (복수)
up_B = st.sidebar.file_uploader("B) 산업용 상세(.parquet, 복수 가능)",
                                type=["parquet"], accept_multiple_files=True)

# 레포 자동탐색
repo_files = find_repo_parquets()
repo_A = [p for p in repo_files if "상품별판매량" in os.path.basename(p) or "A" in os.path.basename(p)]
repo_B = [p for p in repo_files if p not in repo_A]

# A 로딩
if up_A is not None:
    A_raw = read_parquet_any(up_A)
    A_name = up_A.name
elif repo_A:
    A_raw = read_parquet_any(repo_A[0])
    A_name = os.path.basename(repo_A[0])
    st.sidebar.info(f"A 자동 사용: **{A_name}**")
else:
    st.error("A(월별 총괄) Parquet를 업로드하거나 레포지토리에 넣어주세요.")
    st.stop()

# B 로딩
B_frames = []
B_used = []
if up_B:
    for f in up_B:
        df = read_parquet_any(f)
        if df is not None and len(df) > 0:
            B_frames.append(df)
            B_used.append(f.name)
elif repo_B:
    for p in repo_B:
        try:
            df = read_parquet_any(p)
            if df is not None and len(df) > 0:
                B_frames.append(df)
                B_used.append(os.path.basename(p))
        except Exception:
            pass

B_raw = pd.concat(B_frames, ignore_index=True) if B_frames else pd.DataFrame()

# ================= 컬럼 매핑(간단) =================
st.sidebar.header("② 컬럼 매핑")
A_cols = [str(c) for c in A_raw.columns]

def _pickA(cands, default=None):
    for k in cands:
        for c in A_cols:
            if k in c:
                return c
    return default or A_cols[0]

c_date = st.sidebar.selectbox("A: 날짜(월)", A_cols, index=A_cols.index(_pickA(["날짜","월","date","DATE"])) if _pickA(["날짜","월","date","DATE"]) in A_cols else 0)
c_home = st.sidebar.selectbox("A: 주택용(또는 취사용)", A_cols, index=A_cols.index(_pickA(["주택","취사용","가정"])) if _pickA(["주택","취사용","가정"]) in A_cols else 1)
c_ind  = st.sidebar.selectbox("A: 산업용 합계", A_cols, index=A_cols.index(_pickA(["산업"])) if _pickA(["산업"]) in A_cols else 2)

A = A_raw.copy()
A["날짜"]   = pd.to_datetime(A[c_date], errors="coerce")
A["주택용"] = A[c_home].apply(to_num)
A["산업용"] = A[c_ind].apply(to_num)

if not B_raw.empty:
    B_cols = [str(c) for c in B_raw.columns]
    def _pickB(cands, default=None):
        for k in cands:
            for c in B_cols:
                if k in c:
                    return c
        return default or B_cols[0]

    b_date = st.sidebar.selectbox("B: 날짜(월)", B_cols, index=B_cols.index(_pickB(["년월","월","날짜","date"])) if _pickB(["년월","월","날짜","date"]) in B_cols else 0)
    b_indu = st.sidebar.selectbox("B: 업종", B_cols, index=B_cols.index(_pickB(["업종","산업"])) if _pickB(["업종","산업"]) in B_cols else 0)
    b_cust = st.sidebar.selectbox("B: 고객명", B_cols, index=B_cols.index(_pickB(["고객","고객명","거래처"])) if _pickB(["고객","고객명","거래처"]) in B_cols else 0)
    b_amt  = st.sidebar.selectbox("B: 사용량", B_cols, index=B_cols.index(_pickB(["사용","수량","m3","MQ","MJ","Nm3","NM3"])) if _pickB(["사용","수량","m3","MQ","MJ","Nm3","NM3"]) in B_cols else 0)

    def _parse_month(x):
        s = str(x)
        for fmt in ("%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"):
            try:
                return pd.to_datetime(s, format=fmt).replace(day=1)
            except Exception:
                pass
        return pd.to_datetime(s, errors="coerce")

    B = B_raw.copy()
    B["날짜"]  = pd.to_datetime(B[b_date].map(_parse_month), errors="coerce")
    B["업종"]  = B[b_indu].astype(str).str.strip()
    B["고객"]  = B[b_cust].astype(str).str.strip()
    B["사용량"] = pd.to_numeric(B[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
else:
    B = pd.DataFrame(columns=["날짜","업종","고객","사용량"])

# ================= 기간 설정 =================
st.title("📊 도시가스 판매량 분석")
date_min = (min(A["날짜"].min(), B["날짜"].min()) if not B.empty else A["날짜"].min())
date_max = (max(A["날짜"].max(), B["날짜"].max()) if not B.empty else A["날짜"].max())
d1, d2 = st.sidebar.date_input("③ 기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ================= 탭 =================
tab0, tab1, tab2 = st.tabs(["🏠 대시보드","📚 집계","🏭 산업용 집중분석"])

# ---- 탭0
with tab0:
    st.subheader("연도별 용도 누적 스택")
    AA = A[(A["날짜"]>=pd.to_datetime(d1)) & (A["날짜"]<=pd.to_datetime(d2))].copy()
    AA["연도"] = AA["날짜"].dt.year
    annual = AA.groupby("연도", as_index=False)[["주택용","산업용"]].sum().sort_values("연도")

    fig0 = go.Figure()
    for col in ["주택용","산업용"]:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(barmode="stack", template="simple_white", height=420,
                       font=dict(family=FONT, size=13))
    st.plotly_chart(fig0, use_container_width=True)
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ---- 탭1
with tab1:
    st.subheader("집계 — 월/분기/반기/연간")
    gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
    AA = A[(A["날짜"]>=pd.to_datetime(d1)) & (A["날짜"]<=pd.to_datetime(d2))].copy()
    AA["Period"] = as_period_key(AA["날짜"], gran)
    sum_tbl = AA.groupby("Period", as_index=False)[["주택용","산업용"]].sum().sort_values("Period")

    left, right = st.columns([2,3])
    with left:
        st.dataframe(sum_tbl.style.format({"주택용":"{:,.0f}","산업용":"{:,.0f}"}), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["주택용"], name="주택용"))
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["산업용"], name="산업용"))
        fig.update_layout(barmode="group", template="simple_white", height=360,
                          font=dict(family=FONT, size=13))
        st.plotly_chart(fig, use_container_width=True)

# ---- 탭2
with tab2:
    st.subheader("산업용 집중분석 — 업종×기간 히트맵")
    if B.empty:
        st.info("B(산업용 상세) 데이터가 없어 히트맵은 표시하지 않습니다.")
    else:
        gran_f = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_f")
        BB = B[(B["날짜"]>=pd.to_datetime(d1)) & (B["날짜"]<=pd.to_datetime(d2))].copy()
        BB["Period"] = as_period_key(BB["날짜"], gran_f)
        piv = BB.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
        piv = piv[piv.columns.sort_values()].sort_index()

        Z = piv.values; X = piv.columns.tolist(); Y = piv.index.tolist()
        heat = go.Figure(data=go.Heatmap(
            z=Z, x=X, y=Y, colorscale="Blues", colorbar=dict(title="사용량"),
            hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(template="simple_white", height=560,
                           font=dict(family=FONT, size=13),
                           margin=dict(l=70,r=20,t=40,b=40))
        st.plotly_chart(heat, use_container_width=True)

# ---- 사용 파일 요약
with st.expander("🔎 사용한 원천 파일"):
    st.write(f"A: **{A_name}**")
    if B_used:
        st.write("B: " + ", ".join(B_used[:10]) + (" …" if len(B_used) > 10 else ""))
    else:
        st.write("B: (없음)")
