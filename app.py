# app.py — Gas Sales Analytics (Landing + Aggregations + Industrial Focus)
# - 탭3: [산업용 집중분석] 업종×기간 히트맵 → 셀 클릭: 고객 Top-N / YoY / 다운로드
# - Parquet 우선, 업로드 없으면 저장소에 있는 파일 자동 탐색

import os, glob, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# plotly_events 모듈이 없으면 우회 로직 사용
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ───────────────────────── 공통 유틸 ─────────────────────────
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    if gran == "월":
        return d.dt.to_period("M").astype(str)        # e.g., '2025-09'
    elif gran == "분기":
        return d.dt.to_period("Q").astype(str)        # e.g., '2025Q3'
    elif gran == "반기":
        y = d.dt.year.astype(str)
        h = np.where(d.dt.month <= 6, "H1", "H2")
        return y + h                                   # e.g., '2025H1'
    else:
        return d.dt.year.astype(str)                  # e.g., '2025'

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
                   left_on=key_cols + [period_col], right_on=key_cols + ["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs() > 1e-9,
                           out["증감"] / out["전년동기"] * 100, np.nan)
    return out

def find_first(paths_or_patterns):
    """리스트에 패턴/경로를 넣으면 첫 번째 존재 파일 경로 반환"""
    for p in paths_or_patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            found = sorted(glob.glob(p))
            if found:
                return found[0]
        elif os.path.exists(p):
            return p
    return None

def list_all(patterns):
    out = []
    for pat in patterns:
        out += glob.glob(pat)
    return sorted(set(out))

@st.cache_data(show_spinner=False)
def read_parquet_any(path_or_buf):
    return pd.read_parquet(path_or_buf)

@st.cache_data(show_spinner=False)
def concat_parquets(paths_or_buffers):
    frames = []
    for f in paths_or_buffers:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # 중복 열/이상 열 제거(있다면)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    return pd.DataFrame()

def parse_month_like(s):
    s = str(s)
    for fmt in ["%Y-%m", "%Y/%m", "%Y%m", "%Y.%m", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

# ───────────────────────── 데이터 입력 ─────────────────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종) — Parquet 권장")

# A) 월별 총괄 — 업로드(1개) 또는 자동탐색(상품별판매량.parquet)
up_A = st.sidebar.file_uploader("A) 월별 총괄(.parquet)", type=["parquet"])
if up_A:
    A_raw = read_parquet_any(up_A)
    A_used = up_A.name
else:
    # 저장소 자동
    A_auto = find_first(["상품별판매량.parquet", "overall.parquet"])
    if A_auto:
        A_raw = read_parquet_any(A_auto)
        A_used = os.path.basename(A_auto)
        st.sidebar.info(f"A 자동 사용: **{A_used}**")
    else:
        st.info("A(월별 총괄) Parquet 파일을 업로드하거나, 저장소에 `상품별판매량.parquet`를 두면 자동 인식돼.")
        st.stop()

# A 컬럼 매핑
colsA = A_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")

def pickA(keys, default=None):
    for k in keys:
        for c in colsA:
            if k in c:
                return c
    return default if default else colsA[0]

a_date = st.sidebar.selectbox("날짜(월)", colsA, index=colsA.index(pickA(["날짜", "월", "Date"])) if pickA(["날짜","월","Date"]) in colsA else 0)
# 주택/산업 합산용 최소 필드(원하면 아래에 더 추가 가능)
a_home = st.sidebar.selectbox("주택용 합계 컬럼", colsA, index=colsA.index(pickA(["주택용","취사용"])) if pickA(["주택용","취사용"]) in colsA else 0)
a_ind  = st.sidebar.selectbox("산업용 합계 컬럼", colsA, index=colsA.index(pickA(["산업용"])) if pickA(["산업용"]) in colsA else 0)

A = A_raw.copy()
A["날짜"] = pd.to_datetime(A[a_date], errors="coerce")
A["주택용"] = to_num(A[a_home])
A["산업용"] = to_num(A[a_ind])

# B) 산업용 상세 — 업로드(여러개) 또는 자동탐색(가정용외_*.parquet, *_산업*.parquet 등)
up_B = st.sidebar.file_uploader("B) 산업용 상세(.parquet, 여러 개 가능)", type=["parquet"], accept_multiple_files=True)
B_used = []

if up_B:
    B = concat_parquets(up_B)
    B_used = [f.name for f in up_B]
else:
    pats = [
        "가정용외_*.parquet", "*산업*상세*.parquet", "*산업용*.parquet"
    ]
    files = list_all(pats)
    if files:
        B = concat_parquets(files)
        B_used = [os.path.basename(x) for x in files]
        st.sidebar.info("B 자동 병합: " + ", ".join(B_used[:6]) + (" …" if len(B_used) > 6 else ""))
    else:
        # 없어도 나머지 탭은 동작하도록 빈 프레임
        B = pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

colsB = B.columns.astype(str).tolist()
st.sidebar.header("③ B(산업용 상세) 컬럼 매핑")

def pickB(keys, default=None):
    for k in keys:
        for c in colsB:
            if k in c:
                return c
    return default if default else (colsB[0] if colsB else None)

if len(colsB) > 0:
    b_date = st.sidebar.selectbox("날짜(월/일)", colsB, index=(colsB.index(pickB(["청구년월","사용월","년월","월","날짜","일자"])) if pickB(["청구년월","사용월","년월","월","날짜","일자"]) in colsB else 0))
    b_use  = st.sidebar.selectbox("용도", colsB, index=(colsB.index(pickB(["용도"])) if pickB(["용도"]) in colsB else 0))
    b_ind  = st.sidebar.selectbox("업종", colsB, index=(colsB.index(pickB(["업종"])) if pickB(["업종"]) in colsB else 0))
    b_cus  = st.sidebar.selectbox("고객명", colsB, index=(colsB.index(pickB(["고객","고객명","거래처"])) if pickB(["고객","고객명","거래처"]) in colsB else 0))
    b_amt  = st.sidebar.selectbox("사용량 컬럼", colsB, index=(colsB.index(pickB(["사용량","m3","NM3","Nm3","MJ"])) if pickB(["사용량","m3","NM3","Nm3","MJ"]) in colsB else 0))

    Bn = B.copy()
    Bn["날짜"]   = pd.to_datetime(Bn[b_date].apply(parse_month_like), errors="coerce")
    Bn["용도"]   = Bn[b_use].astype(str).str.strip()
    Bn["업종"]   = Bn[b_ind].astype(str).str.strip()
    Bn["고객명"] = Bn[b_cus].astype(str).str.strip()
    Bn["사용량"] = pd.to_numeric(Bn[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
else:
    Bn = B.copy()

# ───────────────────────── 범위/단위 ─────────────────────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")

minA = pd.to_datetime(A["날짜"]).min()
maxA = pd.to_datetime(A["날짜"]).max()
if len(Bn) > 0 and "날짜" in Bn:
    minB = pd.to_datetime(Bn["날짜"]).min()
    maxB = pd.to_datetime(Bn["날짜"]).max()
    dmin = min(minA, minB)
    dmax = max(maxA, maxB)
else:
    dmin, dmax = minA, maxA

d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(dmin), pd.to_datetime(dmax)])

# ───────────────────────── 탭 구성 ─────────────────────────
tab0, tab1, tab2 = st.tabs(["🏠 대시보드", "📚 집계", "🏭 산업용 집중분석"])

# ── 탭0: 랜딩(연도×용도 스택)
with tab0:
    st.subheader("연도별 용도 누적 스택")
    landing = A[(A["날짜"] >= pd.to_datetime(d1)) & (A["날짜"] <= pd.to_datetime(d2))].copy()
    landing["연도"] = landing["날짜"].dt.year
    usage_cols = ["주택용","산업용"]
    annual = landing.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")

    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(
        barmode="stack", template="simple_white", height=420,
        font=dict(family=FONT, size=13), legend=dict(orientation="h", y=1.02, x=0)
    )
    st.plotly_chart(fig0, use_container_width=True, config={"displaylogo": False})
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1: 집계(월/분기/반기/연간)
with tab1:
    st.subheader("집계 — 월/분기/반기/연간 (주택용 / 산업용)")
    gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
    A1 = A[(A["날짜"] >= pd.to_datetime(d1)) & (A["날짜"] <= pd.to_datetime(d2))].copy()
    A1["Period"] = as_period_key(A1["날짜"], gran)
    sum_tbl = A1.groupby("Period", as_index=False)[["주택용","산업용"]].sum().sort_values("Period")

    c1, c2 = st.columns([2,3])
    with c1:
        st.dataframe(sum_tbl.style.format({"주택용":"{:,.0f}","산업용":"{:,.0f}"}), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["주택용"], name="주택용"))
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["산업용"], name="산업용"))
        fig.update_layout(
            barmode="group", template="simple_white", height=360,
            xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
            font=dict(family=FONT, size=13)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── 탭2: 산업용 집중분석(히트맵 → 셀 클릭: 고객 Top-N)
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    if len(Bn) == 0 or "사용량" not in Bn.columns or "업종" not in Bn.columns or "날짜" not in Bn.columns:
        st.info("산업용 상세(B)가 없거나 필수 컬럼(날짜/업종/사용량)이 부족해. 사이드바에서 B를 매핑해줘.")
    else:
        # 산업용만 필터(파일에 다른 용도가 섞여 있을 가능성)
        B2 = Bn.copy()
        if "용도" in B2.columns:
            B2 = B2[B2["용도"].astype(str).str.contains("산업", na=False)]

        if len(B2) == 0:
            st.info("선택한 기간/필터에 산업용 데이터가 없어.")
        else:
            gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
            B2 = B2[(B2["날짜"] >= pd.to_datetime(d1)) & (B2["날짜"] <= pd.to_datetime(d2))].copy()
            B2["Period"] = as_period_key(B2["날짜"], gran_focus)

            piv = B2.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
            piv = piv[piv.columns.sort_values()].sort_index()
            Z = piv.values; X = piv.columns.tolist(); Y = piv.index.tolist()
            zmid = float(np.nanmean(Z)) if np.isfinite(Z).all() else None

            heat = go.Figure(data=go.Heatmap(
                z=Z, x=X, y=Y, colorscale="Blues", zmid=zmid, colorbar=dict(title="사용량"),
                text=piv.round(0).astype(int).astype(str), texttemplate="%{text}", textfont={"size":10},
                hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
            ))
            heat.update_layout(template="simple_white", height=560,
                               xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                               font=dict(family=FONT, size=13), margin=dict(l=70, r=20, t=40, b=40))

            # 클릭 처리(모듈 없으면 드롭다운 대체)
            clicked_period, clicked_ind = None, None
            if HAS_PLOTLY_EVENTS:
                ev = plotly_events(heat, click_event=True, hover_event=False,
                                   select_event=False, override_height=560, override_width="100%")
                if ev:
                    clicked_period = str(ev[0].get("x"))
                    clicked_ind    = str(ev[0].get("y"))
            else:
                st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
                cA, cB = st.columns(2)
                with cA:
                    clicked_period = st.selectbox("기간 선택", X)
                with cB:
                    clicked_ind = st.selectbox("업종 선택", Y)

            if clicked_period and clicked_ind:
                st.markdown(f"**선택 업종:** `{clicked_ind}` · **선택 기간:** `{clicked_period}`")
                yo = yoy_compare(B2[B2["업종"] == clicked_ind], ["업종","고객명"], "사용량", "Period", gran_focus)
                sel = yo[yo["Period"] == clicked_period].copy().sort_values("사용량", ascending=False)
                sel["사용량"] = sel["사용량"].round(0)
                sel["전년동기"] = sel["전년동기"].round(0)
                sel["증감"] = sel["증감"].round(0)
                sel["YoY(%)"] = sel["YoY(%)"].round(1)

                top_n = st.slider("상위 N", 5, 100, 20, step=5)
                view = sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

                g1, g2 = st.columns([1.3, 1.7])
                with g1:
                    st.dataframe(
                        view.style.format({"사용량":"{:,.0f}","전년동기":"{:,.0f}","증감":"{:+,.0f}","YoY(%)":"{:+,.1f}"}),
                        use_container_width=True, height=520
                    )
                    st.download_button(
                        "⬇️ 고객리스트 CSV",
                        data=view.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{clicked_ind}_{clicked_period}_top{top_n}.csv",
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
            else:
                if HAS_PLOTLY_EVENTS:
                    st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
                st.info("히트맵 셀을 클릭(또는 우측 선택박스에서 기간·업종 지정)하면 아래에 고객 Top-N과 그래프가 표시돼.")

# ───────────────────────── 사용 파일 로그 ─────────────────────────
with st.expander("🔎 분석에 사용된 원천 파일"):
    st.write(f"A(월별 총괄): **{A_used}**")
    if B_used:
        st.write("B(산업용 상세): " + ", ".join(B_used[:10]) + (" …" if len(B_used) > 10 else ""))
    else:
        st.write("B(산업용 상세): (업로드/자동탐색 결과 없음)")
