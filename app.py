# app.py — Gas Sales Analytics (월/분기/반기/연간 + 산업용 업종/고객 히트맵)
# - 파일형식: A/B 모두 .xlsx / .csv / .parquet 지원 (openpyxl/pyarrow 필요)
# - B(산업용 상세)는 여러 파일 업로드 또는 저장소 내 자동병합(가정용외_*.{csv,xlsx,parquet})

import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# plotly chart 클릭 이벤트 모듈(있으면 사용, 없으면 드롭다운 대체)
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False
    def plotly_events(*args, **kwargs):
        return []

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"


# ───────────── 유틸 ─────────────
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "").replace(" ", "")
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

def yoy_compare(df, key_cols, value_col, period_col, prev_map):
    """선택 기간의 전년동기/증감/YoY% 계산"""
    gran = st.session_state.get("gran_focus", st.session_state.get("granularity", "월"))
    lag = prev_map.get(gran, 12)
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
def read_excel_any(path_or_buf):
    # openpyxl 사용
    return pd.read_excel(path_or_buf, engine="openpyxl")

@st.cache_data(show_spinner=False)
def read_csv_any(path_or_buf):
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path_or_buf, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path_or_buf, encoding_errors="ignore")

@st.cache_data(show_spinner=False)
def read_parquet_any(path_or_buf):
    return pd.read_parquet(path_or_buf)

def list_existing(patterns):
    out = []
    for pat in patterns:
        out += glob.glob(pat)
    return sorted(set(out))


# ───────────── 사이드바: 입력 ─────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산) · B: 산업용 상세(고객/업종) — CSV/Parquet 권장, 엑셀 허용")

# A 업로더 (xlsx/csv/parquet)
up_overall = st.sidebar.file_uploader("A) 월별 총괄 (xlsx/csv/parquet)", type=["xlsx", "csv", "parquet"])

# A 자동탐색(저장소)
A_CAND = [
    "상품별판매량.parquet", "overall.parquet",
    "상품별판매량.xlsx", "overall.xlsx",
    "상품별판매량.csv", "overall.csv"
]

if up_overall:
    used_overall = up_overall.name
    if used_overall.lower().endswith(".parquet"):
        overall_raw = read_parquet_any(up_overall)
    elif used_overall.lower().endswith(".csv"):
        overall_raw = read_csv_any(up_overall)
    else:
        overall_raw = read_excel_any(up_overall)
else:
    used_overall = next((p for p in A_CAND if os.path.exists(p)), None)
    if used_overall:
        if used_overall.endswith(".parquet"):
            overall_raw = read_parquet_any(used_overall)
        elif used_overall.endswith(".csv"):
            overall_raw = read_csv_any(used_overall)
        else:
            overall_raw = read_excel_any(used_overall)
        st.sidebar.info(f"A 자동 사용: **{used_overall}**")
    else:
        st.warning("A(월별 총괄) 파일(.xlsx/.csv/.parquet)을 업로드하거나 리포에 넣어줘.")
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

c_date   = st.sidebar.selectbox("날짜(월)", colsA, index=colsA.index(_pickA(["날짜","월","Date"])) if _pickA(["날짜","월","Date"]) in colsA else 0)
c_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 0)
c_indh   = st.sidebar.selectbox("개별난방", colsA, index=colsA.index(_pickA(["개별난방"])) if _pickA(["개별난방"]) in colsA else 0)
c_cenh   = st.sidebar.selectbox("중앙난방", colsA, index=colsA.index(_pickA(["중앙난방"])) if _pickA(["중앙난방"]) in colsA else 0)
c_self   = st.sidebar.selectbox("자가열전용", colsA, index=colsA.index(_pickA(["자가열전용","자가열"])) if _pickA(["자가열전용","자가열"]) in colsA else 0)
c_indusA = st.sidebar.selectbox("산업용 합계", colsA, index=colsA.index(_pickA(["산업용"])) if _pickA(["산업용"]) in colsA else 0)

CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","일반용","일반용(1)","일반용(2)"]
extra_present = [c for c in CAND_EXTRA if c in colsA]
extra_selects = {}
if extra_present:
    st.sidebar.markdown("**추가 용도(선택)**")
    for nm in extra_present:
        extra_selects[nm] = st.sidebar.selectbox(nm, colsA, index=colsA.index(nm))

# A 정리
overall = overall_raw.copy()
overall["날짜"]    = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"]  = overall[c_cook].apply(to_num)
overall["개별난방"] = overall[c_indh].apply(to_num)
overall["중앙난방"] = overall[c_cenh].apply(to_num)
overall["자가열전용"] = overall[c_self].apply(to_num)
overall["산업용"]   = overall[c_indusA].apply(to_num)
overall["주택용"]   = overall[["취사용","개별난방","중앙난방","자가열전용"]].sum(axis=1)
for nm, col in extra_selects.items():
    overall[nm] = overall[col].apply(to_num)

# B 업로더 (여러 파일)
up_indetail = st.sidebar.file_uploader("B) 산업용 상세(여러개 업로드 가능) — xlsx/csv/parquet",
                                       type=["xlsx","csv","parquet"], accept_multiple_files=True)

# B 자동 탐색
B_FILES = list_existing(["가정용외_*.parquet","가정용외_*.csv","가정용외_*.xlsx"])

if up_indetail:
    used_inds = [f.name for f in up_indetail]
    frames = []
    for f in up_indetail:
        if f.name.lower().endswith(".parquet"): frames.append(read_parquet_any(f))
        elif f.name.lower().endswith(".csv"):   frames.append(read_csv_any(f))
        else:                                   frames.append(read_excel_any(f))
    indetail_raw = pd.concat(frames, ignore_index=True)
elif B_FILES:
    used_inds = [os.path.basename(p) for p in B_FILES]
    frames = []
    for p in B_FILES:
        if p.endswith(".parquet"): frames.append(read_parquet_any(p))
        elif p.endswith(".csv"):   frames.append(read_csv_any(p))
        else:                      frames.append(read_excel_any(p))
    indetail_raw = pd.concat(frames, ignore_index=True)
    st.sidebar.info("B 자동 병합: " + ", ".join(used_inds[:6]) + (" …" if len(used_inds)>6 else ""))
else:
    used_inds = []
    indetail_raw = pd.DataFrame(columns=["청구년월","용도","업종","고객명","사용량"])

# B 컬럼 매핑
colsB = indetail_raw.columns.astype(str).tolist()
st.sidebar.header("③ B(산업용 상세) 컬럼 매핑")

def _pickB(keys, default=None):
    for k in keys:
        for c in colsB:
            if k in c:
                return c
    return default if default else (colsB[0] if colsB else None)

b_date = st.sidebar.selectbox("날짜(월)", colsB, index=(colsB.index(_pickB(["청구년월","사용월","년월","월"])) if _pickB(["청구년월","사용월","년월","월"]) in colsB else 0)) if colsB else None
b_use  = st.sidebar.selectbox("용도", colsB, index=(colsB.index(_pickB(["용도"])) if _pickB(["용도"]) in colsB else 0)) if colsB else None
b_ind  = st.sidebar.selectbox("업종", colsB, index=(colsB.index(_pickB(["업종"])) if _pickB(["업종"]) in colsB else 0)) if colsB else None
b_cus  = st.sidebar.selectbox("고객명", colsB, index=(colsB.index(_pickB(["고객","고객명","거래처"])) if _pickB(["고객","고객명","거래처"]) in colsB else 0)) if colsB else None
# 사용량 컬럼 후보 다양화
b_amt  = st.sidebar.selectbox("사용량 열", colsB, index=(colsB.index(_pickB(["사용량(m3","Nm3","NM3","m3사용량","사용량","수량","MJ"])) if _pickB(["사용량(m3","Nm3","NM3","m3사용량","사용량","수량","MJ"]) in colsB else 0)) if colsB else None

def parse_month_any(x):
    s = str(x)
    for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt).replace(day=1)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

if colsB:
    indetail = indetail_raw.copy()
    indetail["날짜"]   = pd.to_datetime(indetail[b_date].apply(parse_month_any), errors="coerce")
    indetail["용도"]   = indetail[b_use].astype(str).str.strip()
    indetail["업종"]   = indetail[b_ind].astype(str).str.strip()
    indetail["고객명"] = indetail[b_cus].astype(str).str.strip()
    indetail["사용량"] = pd.to_numeric(indetail[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
else:
    indetail = pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

# ───────────── 공통 기간 선택 ─────────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")
date_min = min(overall["날짜"].min(), indetail["날짜"].min()) if len(indetail)>0 else overall["날짜"].min()
date_max = max(overall["날짜"].max(), indetail["날짜"].max()) if len(indetail)>0 else overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# ───────────── 탭 ─────────────
tab0, tab1, tab2 = st.tabs(["🏠 대시보드", "📚 집계", "🏭 산업용 집중분석"])

# ── 탭0: 연도별 용도 누적 스택
with tab0:
    st.subheader("연도별 용도 누적 스택")
    A = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
    A["연도"] = A["날짜"].dt.year
    usage_cols = ["주택용","산업용"] + [c for c in CAND_EXTRA if c in A.columns]
    annual = A.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")

    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col, text=[f"{v:,.0f}" for v in annual[col]], textposition="outside"))
    fig0.update_layout(barmode="stack", template="simple_white", height=420,
                       xaxis=dict(title="Year"), yaxis=dict(title="사용량"),
                       font=dict(family=FONT, size=13),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig0, use_container_width=True, config={"displaylogo": False})

    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1: 집계표/그래프
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
        fig.update_layout(barmode="group", template="simple_white", height=360,
                          xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
                          font=dict(family=FONT, size=13))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── 탭2: 산업용 집중분석
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    if len(indetail) == 0:
        st.info("산업용 상세(B) 데이터가 없어. 업로드하거나 저장소에 가정용외_* 파일을 넣어줘.")
    else:
        gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
        B = indetail[(indetail["날짜"]>=pd.to_datetime(d1)) & (indetail["날짜"]<=pd.to_datetime(d2))].copy()
        if "용도" in B.columns:
            B = B[B["용도"].str.contains("산업", na=False)]

        B["Period"] = as_period_key(B["날짜"], gran_focus)

        # ① 업종×기간 피벗 → 히트맵
        pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
        pivot = pivot[pivot.columns.sort_values()].sort_index()
        Z = pivot.values
        X = pivot.columns.tolist()
        Y = pivot.index.tolist()
        text_labels = [[f"{int(v):,}" if np.isfinite(v) else "" for v in row] for row in Z]

        heat = go.Figure(data=go.Heatmap(
            z=Z, x=X, y=Y, colorscale="Blues", colorbar=dict(title="사용량"),
            text=text_labels, texttemplate="%{text}", hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(template="simple_white", height=560,
                           xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                           font=dict(family=FONT, size=13), margin=dict(l=70,r=20,t=40,b=40))

        clicked = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                                override_height=560, override_width="100%") if HAS_PLOTLY_EVENTS else []

        st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False}) if not HAS_PLOTLY_EVENTS else None

        # ② 셀 선택 (클릭 또는 드롭다운 대체)
        if clicked:
            sel_period = X[clicked[0]["x"]]
            sel_ind = Y[clicked[0]["y"]]
        else:
            c1, c2 = st.columns(2)
            sel_ind = c1.selectbox("업종 선택", Y) if len(Y) else None
            sel_period = c2.selectbox("기간 선택", X) if len(X) else None

        if sel_ind and sel_period:
            st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")
            prev_map = {"월":12, "분기":4, "반기":2, "연간":1}
            yo = yoy_compare(B[B["업종"]==sel_ind], ["업종","고객명"], "사용량", "Period", prev_map)
            yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)

            yo_sel["사용량"]   = yo_sel["사용량"].round(0)
            yo_sel["전년동기"] = yo_sel["전년동기"].round(0)
            yo_sel["증감"]     = yo_sel["증감"].round(0)
            yo_sel["YoY(%)"]   = yo_sel["YoY(%)"].round(1)

            top_n = st.slider("상위 N", 5, 100, 20, step=5)
            view = yo_sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

            g1, g2 = st.columns([1.4,1.6])
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
                fig_bar.update_layout(template="simple_white", height=520,
                                      xaxis=dict(title="고객명", tickangle=-45),
                                      yaxis=dict(title="사용량"),
                                      font=dict(family=FONT, size=12), margin=dict(l=40,r=20,t=10,b=120))
                st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("히트맵 셀을 클릭하거나 위 드롭다운으로 업종·기간을 선택해줘.")

# ───────────── 사용 파일 표시 ─────────────
with st.expander("🔎 분석에 사용된 원천 파일"):
    if used_overall: st.write(f"A(월별 총괄): **{used_overall}**")
    if len(used_inds): st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds)>10 else ""))
