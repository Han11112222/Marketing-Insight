# app.py — Gas Sales Analytics (fast I/O + Parquet-first + Heatmap labels)
# 탭3: [산업용 집중분석] 업종×기간 히트맵 → (선택) 셀 클릭 시 고객 Top-N / YoY / 다운로드

import os, glob, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── (선택) 클릭 이벤트 모듈: 없으면 읽기 전용
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ▒▒ 공통 유틸 ▒▒
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
           .groupby(key_cols + ["_prev"], as_index=False)[value_col].sum()
           .rename(columns={value_col: "전년동기"})
    )
    out = pd.merge(a, b, how="left", left_on=key_cols + [period_col], right_on=key_cols + ["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs() > 1e-9, out["증감"] / out["전년동기"] * 100, np.nan)
    return out

# ▒▒ 파일 읽기 (Parquet 우선, 캐시) ▒▒
@st.cache_data(show_spinner=False, ttl=600)
def read_parquet_fast(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

@st.cache_data(show_spinner=False, ttl=600)
def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, encoding_errors="ignore", low_memory=False)

@st.cache_data(show_spinner=False, ttl=600)
def read_excel_any(path: str | bytes):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

def find_first(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def list_existing(patterns):
    out = []
    for pat in patterns:
        out += glob.glob(pat)
    return sorted(set(out))

# ▒▒ 사이드바: 데이터 입력 ▒▒
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산) · B: 산업용 상세(고객/업종) — Parquet 권장")

# A) 월별 총괄
up_overall = st.sidebar.file_uploader("A) 월별 총괄 (parquet/csv/xlsx)", type=["parquet", "csv", "xlsx"])
used_overall = None

if up_overall is not None:
    if up_overall.name.lower().endswith(".parquet"):
        overall_raw = read_parquet_fast(up_overall)
    elif up_overall.name.lower().endswith(".csv"):
        overall_raw = pd.read_csv(up_overall, encoding="utf-8-sig")
    else:
        overall_raw = read_excel_any(up_overall)
    used_overall = up_overall.name
else:
    # 깃허브/로컬에 있는 파일 자동 탐색 (Parquet 우선)
    used_overall = find_first([
        "상품별판매량.parquet", "상품별판매량.csv", "상품별판매량.xlsx",
        "월별총괄.parquet", "월별총괄.csv", "월별총괄.xlsx",
        "overall.parquet", "overall.csv", "overall.xlsx"
    ])
    if used_overall:
        if used_overall.lower().endswith(".parquet"):
            overall_raw = read_parquet_fast(used_overall)
        elif used_overall.lower().endswith(".csv"):
            overall_raw = read_csv_any(used_overall)
        else:
            overall_raw = read_excel_any(used_overall)
        st.sidebar.info(f"A 자동 사용: **{used_overall}**")
    else:
        st.warning("A(월별 총괄) 파일이 필요해요. `상품별판매량.parquet/csv/xlsx` 중 하나를 업로드/저장해 주세요.")
        st.stop()

# A 컬럼 매핑
colsA = overall_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")

def _pickA(keys, default_idx=0):
    for k in keys:
        for c in colsA:
            if k in c:
                return c
    return colsA[default_idx]

c_date   = st.sidebar.selectbox("날짜", colsA, index=colsA.index(_pickA(["날짜", "Date", "월"])) if _pickA(["날짜","Date","월"]) in colsA else 0)
c_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 1)
c_indh   = st.sidebar.selectbox("개별난방", colsA, index=colsA.index(_pickA(["개별난방"])) if _pickA(["개별난방"]) in colsA else 2)
c_cenh   = st.sidebar.selectbox("중앙난방", colsA, index=colsA.index(_pickA(["중앙난방"])) if _pickA(["중앙난방"]) in colsA else 3)
c_self   = st.sidebar.selectbox("자가열전용", colsA, index=colsA.index(_pickA(["자가열전용","자가열"])) if _pickA(["자가열전용","자가열"]) in colsA else 4)
c_indusA = st.sidebar.selectbox("산업용 합계", colsA, index=colsA.index(_pickA(["산업용"])) if _pickA(["산업용"]) in colsA else 5)

# 추가 용도(있으면 자동 추가)
CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","열병합용1","열병합용2","일반용","일반용(1)","일반용(2)"]
extra_present = [c for c in CAND_EXTRA if c in colsA]
extra_selects = {}
if extra_present:
    st.sidebar.markdown("**추가 용도(선택적)**")
    for nm in extra_present:
        extra_selects[nm] = st.sidebar.selectbox(nm, colsA, index=colsA.index(nm))

overall = overall_raw.copy()
overall["날짜"] = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"] = overall[c_cook].apply(to_num)
overall["개별난방"] = overall[c_indh].apply(to_num)
overall["중앙난방"] = overall[c_cenh].apply(to_num)
overall["자가열전용"] = overall[c_self].apply(to_num)
overall["산업용"] = overall[c_indusA].apply(to_num)
overall["주택용"] = overall[["취사용","개별난방","중앙난방","자가열전용"]].sum(axis=1)
for nm, col in extra_selects.items():
    overall[nm] = overall[col].apply(to_num)

# B) 산업용 상세 (여러 파일 업로드/자동 병합, Parquet 우선)
st.sidebar.header("③ B(산업용 상세) 업로드")
up_indetail = st.sidebar.file_uploader("B) 산업용 상세 — 여러 개 가능(parquet/csv/xlsx)", type=["parquet","csv","xlsx","xls"], accept_multiple_files=True)
used_inds = []
if up_indetail:
    frames = []
    for f in up_indetail:
        used_inds.append(f.name)
        if f.name.lower().endswith(".parquet"):
            frames.append(read_parquet_fast(f))
        elif f.name.lower().endswith(".csv"):
            frames.append(pd.read_csv(f, encoding="utf-8-sig"))
        else:
            frames.append(pd.read_excel(f))
    indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
else:
    files = list_existing(["가정용외_*.parquet","가정용외_*.csv","가정용외_*.xlsx","가정용외_*.xls"])
    if files:
        used_inds = [os.path.basename(p) for p in files]
        frames = []
        for p in files:
            if p.lower().endswith(".parquet"):
                frames.append(read_parquet_fast(p))
            elif p.lower().endswith(".csv"):
                frames.append(read_csv_any(p))
            else:
                frames.append(read_excel_any(p))
        indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.sidebar.info("B 자동 병합: " + ", ".join(used_inds[:6]) + (" …" if len(used_inds) > 6 else ""))
    else:
        indetail_raw = pd.DataFrame(columns=["청구년월","용도","업종","고객명","사용량"])

# B 컬럼 매핑
colsB = indetail_raw.columns.astype(str).tolist()
st.sidebar.header("④ B(산업용 상세) 컬럼 매핑")

def _pickB(keys, default=None):
    for k in keys:
        for c in colsB:
            if k in c:
                return c
    return default if default else (colsB[0] if colsB else None)

b_date = st.sidebar.selectbox("날짜(월)", colsB, index=(colsB.index(_pickB(["청구년월","사용월","년월"])) if _pickB(["청구년월","사용월","년월"]) in colsB else 0)) if len(colsB)>0 else None
b_use  = st.sidebar.selectbox("용도", colsB, index=(colsB.index(_pickB(["용도"])) if _pickB(["용도"]) in colsB else 0)) if len(colsB)>0 else None
b_ind  = st.sidebar.selectbox("업종", colsB, index=(colsB.index(_pickB(["업종"])) if _pickB(["업종"]) in colsB else 0)) if len(colsB)>0 else None
b_cus  = st.sidebar.selectbox("고객명", colsB, index=(colsB.index(_pickB(["고객","고객명","거래처"])) if _pickB(["고객","고객명","거래처"]) in colsB else 0)) if len(colsB)>0 else None
b_amt  = st.sidebar.selectbox("사용량 열", colsB, index=(colsB.index(_pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"])) if _pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"]) in colsB else 0)) if len(colsB)>0 else None

def parse_month(x):
    s = str(x)
    for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt).replace(day=1)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

if len(colsB) > 0:
    indetail = indetail_raw.copy()
    indetail["날짜"] = pd.to_datetime(indetail[b_date].apply(parse_month), errors="coerce")
    indetail["용도"] = indetail[b_use].astype(str).str.strip()
    indetail["업종"] = indetail[b_ind].astype(str).str.strip()
    indetail["고객명"] = indetail[b_cus].astype(str).str.strip()
    indetail["사용량"] = pd.to_numeric(
        indetail[b_amt].astype(str).str.replace(",","").str.replace(" ",""),
        errors="coerce"
    ).fillna(0)
else:
    indetail = pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

# ▒▒ 범위/단위 ▒▒
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")
date_min = min(overall["날짜"].min(), indetail["날짜"].min()) if len(indetail)>0 else overall["날짜"].min()
date_max = max(overall["날짜"].max(), indetail["날짜"].max()) if len(indetail)>0 else overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ▒▒ 탭 ▒▒
tab0, tab1, tab2 = st.tabs(["🏠 대시보드","📚 집계","🏭 산업용 집중분석"])

# ── 탭0: 연도×용도 스택
with tab0:
    st.subheader("연도별 용도 누적 스택")
    landing = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
    landing["연도"] = landing["날짜"].dt.year
    usage_cols = ["주택용","산업용"] + [c for c in CAND_EXTRA if c in overall.columns]
    annual = landing.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")
    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(
        barmode="stack", template="simple_white", height=420,
        xaxis=dict(title="Year"), yaxis=dict(title="사용량"),
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    st.plotly_chart(fig0, use_container_width=True, config={"displaylogo": False})
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1: 집계
with tab1:
    st.subheader("집계 — 월/분기/반기/연간 (주택용 / 산업용)")
    gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
    A = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
    A["Period"] = as_period_key(A["날짜"], gran)
    sum_tbl = A.groupby("Period", as_index=False)[["주택용","산업용"]].sum().sort_values("Period")
    left, right = st.columns([2,3])
    with left:
        st.dataframe(sum_tbl.style.format({"주택용":"{:,.0f}","산업용":"{:,.0f}"}), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["주택용"], name="주택용"))
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["산업용"], name="산업용"))
        fig.update_layout(
            barmode="group", template="simple_white", height=360,
            xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
            font=dict(family=FONT, size=13)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── 탭2: 산업용 집중분석
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    if len(indetail) == 0:
        st.info("산업용 상세 파일(B)이 없어 히트맵을 표시할 수 없어.")
        st.stop()

    gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
    B = indetail[(indetail["날짜"]>=pd.to_datetime(d1)) & (indetail["날짜"]<=pd.to_datetime(d2))].copy()

    if "용도" in B.columns:
        B = B[B["용도"].astype(str).str.contains("산업", na=False)]

    if len(B) == 0:
        st.info("선택한 기간/필터에 산업용 데이터가 없어.")
        st.stop()

    B["Period"] = as_period_key(B["날짜"], gran_focus)

    # 업종×기간 피벗
    pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
    pivot = pivot[pivot.columns.sort_values()].sort_index()
    Z = pivot.values
    X = pivot.columns.tolist()
    Y = pivot.index.tolist()
    zmid = float(np.nanmean(Z)) if np.isfinite(Z).all() else None

    # 히트맵 + 텍스트 라벨(업종명: 값 축약)
    # 표시는 k 단위로 축약
    def fmt_k(v):
        try:
            v = float(v)
            if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f}B"
            if v >= 1_000_000:     return f"{v/1_000_000:.1f}M"
            if v >= 1_000:         return f"{v/1_000:.1f}k"
            return f"{v:.0f}"
        except:
            return ""

    text = np.vectorize(fmt_k)(Z)

    heat = go.Figure(data=go.Heatmap(
        z=Z, x=X, y=Y, colorscale="Blues", zmid=zmid,
        colorbar=dict(title="사용량"),
        text=text, texttemplate="%{text}", textfont={"size":11},
        hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
    ))
    heat.update_layout(
        template="simple_white", height=560,
        xaxis=dict(title="Period"), yaxis=dict(title="업종"),
        font=dict(family=FONT, size=13), margin=dict(l=70, r=20, t=40, b=40)
    )

    if HAS_PLOTLY_EVENTS:
        clicked = plotly_events(
            heat, click_event=True, hover_event=False, select_event=False,
            override_height=560, override_width="100%"
        )
    else:
        st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
        st.caption("※ 셀 클릭 기능을 쓰려면 requirements.txt에 `streamlit-plotly-events`를 추가하세요.")
        clicked = []

    if clicked:
        c = clicked[0]
        sel_period = str(c.get("x"))
        sel_ind    = str(c.get("y"))
        st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")

        yo = yoy_compare(B[B["업종"] == sel_ind], ["업종", "고객명"], "사용량", "Period", gran_focus)
        yo_sel = yo[yo["Period"] == sel_period].copy().sort_values("사용량", ascending=False)

        yo_sel["사용량"]   = yo_sel["사용량"].round(0)
        yo_sel["전년동기"] = yo_sel["전년동기"].round(0)
        yo_sel["증감"]     = yo_sel["증감"].round(0)
        yo_sel["YoY(%)"]  = yo_sel["YoY(%)"].round(1)

        top_n = st.slider("상위 N", 5, 100, 20, step=5)
        view = yo_sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

        g1, g2 = st.columns([1.4, 1.6])
        with g1:
            st.dataframe(
                view.style.format({"사용량":"{:,.0f}","전년동기":"{:,.0f}","증감":"{:+,.0f}","YoY(%)":"{:+,.1f}"}),
                use_container_width=True, height=520
            )
            st.download_button(
                "⬇️ 고객리스트 CSV",
                data=view.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sel_ind}_{sel_period}_top{top_n}.csv",
                mime="text/csv"
            )
        with g2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=view["고객명"], y=view["사용량"], name="사용량",
                text=[f"{v:,.0f}" for v in view["사용량"]], textposition="auto"
            ))
            fig_bar.update_layout(
                template="simple_white", height=520,
                xaxis=dict(title="고객명", tickangle=-45),
                yaxis=dict(title="사용량"),
                font=dict(family=FONT, size=12),
                margin=dict(l=40, r=20, t=10, b=120)
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

# ▒▒ 사용 파일 확인 ▒▒
with st.expander("🔎 분석에 사용된 원천 파일"):
    if used_overall:
        st.write(f"A(월별 총괄): **{used_overall}**")
    if used_inds:
        st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds) > 10 else ""))
