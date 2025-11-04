# app.py — Gas Sales Analytics (최적화 + 산업용 집중분석)
# 탭0: 연도×용도 누적 스택
# 탭1: 월/분기/반기/연간 집계
# 탭2: 산업용 집중분석(업종×기간 히트맵 → 고객 Top-N)
# - CSV/Parquet 권장, Excel도 가능(openpyxl)

import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# (선택) 히트맵 셀 클릭 지원: 패키지 없으면 자동 우회(드롭다운 선택)
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False
    def plotly_events(*args, **kwargs):
        return []

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ───────────────────────── 공통 유틸 ─────────────────────────
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
    gran = st.session_state.get("granularity","월")
    lag = prev_map.get(gran, 12)
    p = df[period_col].astype(str)
    if gran in ["월","분기"]:
        prev = (pd.PeriodIndex(p) - lag).astype(str)
    elif gran == "반기":
        y = p.str[:4].astype(int)
        h = p.str[-2:].map({"H1":1,"H2":2}).astype(int)
        idx = (y - y.min())*2 + (h - 1)
        prev_idx = idx - 2
        base = y.min()
        prev = ((prev_idx//2)+base).astype(str) + np.where((prev_idx%2)==0,"H1","H2")
    else:
        prev = (p.astype(int) - 1).astype(str)

    cur = df.copy()
    cur["_prev"] = prev
    a = cur.groupby(key_cols + [period_col], as_index=False)[value_col].sum()
    b = cur.rename(columns={period_col:"_prev"}) \
           .groupby(key_cols+["_prev"], as_index=False)[value_col].sum() \
           .rename(columns={value_col:"전년동기"})
    out = pd.merge(a, b, how="left",
                   left_on=key_cols+[period_col],
                   right_on=key_cols+["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs()>1e-9, out["증감"]/out["전년동기"]*100, np.nan)
    return out

def _file_cache_key(file_obj_or_path):
    if hasattr(file_obj_or_path, "name") and hasattr(file_obj_or_path, "getbuffer"):
        buf = file_obj_or_path.getbuffer()
        return (file_obj_or_path.name, len(buf))
    elif isinstance(file_obj_or_path, str) and os.path.exists(file_obj_or_path):
        stat = os.stat(file_obj_or_path)
        return (file_obj_or_path, stat.st_size, int(stat.st_mtime))
    else:
        return (str(file_obj_or_path),)

@st.cache_data(show_spinner=True)
def read_excel_any(path_or_buf, usecols=None):
    # engine 고정으로 지연 최소화
    key = (_file_cache_key(path_or_buf), tuple(usecols) if usecols else None, "xlsx")
    return pd.read_excel(path_or_buf, engine="openpyxl", usecols=usecols)

@st.cache_data(show_spinner=True)
def read_csv_any(path_or_buf, usecols=None):
    for enc in ["cp949","euc-kr","utf-8-sig","utf-8"]:
        try:
            df = pd.read_csv(path_or_buf, encoding=enc, usecols=usecols)
            return df
        except Exception:
            pass
    return pd.read_csv(path_or_buf, encoding_errors="ignore", usecols=usecols)

@st.cache_data(show_spinner=True)
def read_parquet_any(path_or_buf, columns=None):
    key = (_file_cache_key(path_or_buf), tuple(columns) if columns else None, "parquet")
    return pd.read_parquet(path_or_buf, columns=columns)

def list_existing(patterns):
    out=[]
    for pat in patterns: out+=glob.glob(pat)
    return sorted(set(out))

def parse_month_like(x):
    s = str(x)
    for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt).replace(day=1)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def optimize_indetail(df):
    need = ["날짜","용도","업종","고객명","사용량"]
    keep = [c for c in need if c in df.columns]
    df = df[keep].copy()
    for c in ["용도","업종","고객명"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "사용량" in df.columns:
        df["사용량"] = pd.to_numeric(df["사용량"], errors="coerce").fillna(0)
    return df

# ───────────────────────── 사이드바 업로드 ─────────────────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종) — CSV/Parquet 권장")

# A) 월별 총괄
up_overall = st.sidebar.file_uploader("A) 월별 총괄(.parquet/.csv)", type=["parquet","csv"])
overall_raw = None
used_overall = None

if up_overall:
    used_overall = up_overall.name
    if used_overall.lower().endswith(".parquet"):
        overall_raw = read_parquet_any(up_overall)
    else:
        overall_raw = read_csv_any(up_overall)
else:
    # 저장소 기본 파일 자동 사용(있는 경우)
    cands = list_existing(["상품별판매량.parquet","상품별판매량.csv","overall.parquet","overall.csv"])
    if cands:
        used_overall = os.path.basename(cands[0])
        if used_overall.lower().endswith(".parquet"):
            overall_raw = read_parquet_any(cands[0])
        else:
            overall_raw = read_csv_any(cands[0])

if overall_raw is None or len(overall_raw)==0:
    st.warning("A 파일이 필요합니다. CSV 또는 Parquet로 업로드/저장해 주세요.")
    st.stop()

# A 컬럼 매핑(간단화: 자동 추론 + 필요시 수정)
colsA = overall_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")

def guessA(keys, default=None):
    for k in keys:
        for c in colsA:
            if k in c:
                return c
    return default if default else colsA[0]

c_date   = st.sidebar.selectbox("날짜(월/일자)", colsA, index=colsA.index(guessA(["날짜","월","Date"])) if guessA(["날짜","월","Date"]) in colsA else 0)
c_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(guessA(["취사용"])) if guessA(["취사용"]) in colsA else 0)
c_indh   = st.sidebar.selectbox("개별난방", colsA, index=colsA.index(guessA(["개별난방"])) if guessA(["개별난방"]) in colsA else 0)
c_cenh   = st.sidebar.selectbox("중앙난방", colsA, index=colsA.index(guessA(["중앙난방"])) if guessA(["중앙난방"]) in colsA else 0)
c_self   = st.sidebar.selectbox("자가열전용", colsA, index=colsA.index(guessA(["자가열전용","자가열"])) if guessA(["자가열전용","자가열"]) in colsA else 0)
c_indusA = st.sidebar.selectbox("산업용 합계", colsA, index=colsA.index(guessA(["산업용"])) if guessA(["산업용"]) in colsA else 0)

CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","일반용(1)","일반용(2)","일반용"]
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

# B) 산업용 상세 — 여러 파일
st.sidebar.header("③ B(산업용 상세) 업로드")
up_indetail = st.sidebar.file_uploader("B) 산업용 상세 — CSV/Parquet/XLSX 여러 개", type=["csv","parquet","xlsx","xls"], accept_multiple_files=True)
used_inds = []
if up_indetail:
    frames=[]
    for f in up_indetail:
        used_inds.append(f.name)
        if f.name.lower().endswith(".parquet"):
            df = read_parquet_any(f)
        elif f.name.lower().endswith(".csv"):
            df = read_csv_any(f)
        else:
            df = read_excel_any(f)
        frames.append(df)
    indetail_raw = pd.concat(frames, ignore_index=True)
else:
    # 저장소 자동 탐색(가정용외_*.parquet/csv/xlsx)
    pats = ["가정용외_*.parquet","가정용외_*.csv","가정용외_*.xlsx","가정용외_*.xls"]
    files = list_existing(pats)
    if files:
        used_inds = [os.path.basename(p) for p in files]
        frames=[]
        for p in files:
            if p.lower().endswith(".parquet"):
                df = read_parquet_any(p)
            elif p.lower().endswith(".csv"):
                df = read_csv_any(p)
            else:
                df = read_excel_any(p)
            frames.append(df)
        indetail_raw = pd.concat(frames, ignore_index=True)
    else:
        indetail_raw = pd.DataFrame(columns=["청구년월","용도","업종","고객명","사용량"])

# B 컬럼 매핑
colsB = indetail_raw.columns.astype(str).tolist()
st.sidebar.header("④ B(산업용 상세) 컬럼 매핑")

def guessB(keys, default=None):
    for k in keys:
        for c in colsB:
            if k in c:
                return c
    return default if default else (colsB[0] if colsB else None)

b_date = st.sidebar.selectbox("날짜(월)", colsB, index=(colsB.index(guessB(["청구년월","사용월","년월","월"])) if guessB(["청구년월","사용월","년월","월"]) in colsB else 0)) if len(colsB)>0 else None
b_use  = st.sidebar.selectbox("용도", colsB, index=(colsB.index(guessB(["용도"])) if guessB(["용도"]) in colsB else 0)) if len(colsB)>0 else None
b_ind  = st.sidebar.selectbox("업종", colsB, index=(colsB.index(guessB(["업종"])) if guessB(["업종"]) in colsB else 0)) if len(colsB)>0 else None
b_cus  = st.sidebar.selectbox("고객명", colsB, index=(colsB.index(guessB(["고객","고객명","거래처"])) if guessB(["고객","고객명","거래처"]) in colsB else 0)) if len(colsB)>0 else None
b_amt  = st.sidebar.selectbox("사용량 열", colsB, index=(colsB.index(guessB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ","사용량(㎥)"])) if guessB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ","사용량(㎥)"]) in colsB else 0)) if len(colsB)>0 else None

if len(colsB)>0:
    indetail = indetail_raw.copy()
    indetail["날짜"] = pd.to_datetime(indetail[b_date].map(parse_month_like), errors="coerce")
    indetail["용도"] = indetail[b_use].astype(str).str.strip()
    indetail["업종"] = indetail[b_ind].astype(str).str.strip()
    indetail["고객명"]= indetail[b_cus].astype(str).str.strip()
    indetail["사용량"]= pd.to_numeric(indetail[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
    indetail = optimize_indetail(indetail)
else:
    indetail = pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

# ───────────────────────── 기간/단위 선택 ─────────────────────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")

date_min = min(overall["날짜"].min(), indetail["날짜"].min()) if len(indetail)>0 else overall["날짜"].min()
date_max = max(overall["날짜"].max(), indetail["날짜"].max()) if len(indetail)>0 else overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

# 히트맵에 표시할 업종 개수(상위 N)
max_inds = st.sidebar.number_input("업종 최대 표시 개수(히트맵)", 10, 200, 60, step=10)

# ───────────────────────── 탭 ─────────────────────────
tab0, tab1, tab2 = st.tabs(["🏠 대시보드","📚 집계","🏭 산업용 집중분석"])

# ── 탭0: 연도 스택
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
    if len(indetail)==0:
        st.info("산업용 상세 파일(B)이 없어 히트맵을 표시할 수 없어.")
    else:
        gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
        B = indetail[(indetail["날짜"]>=pd.to_datetime(d1)) & (indetail["날짜"]<=pd.to_datetime(d2))].copy()

        # 산업용만 필터(파일에 다른 용도 섞여 있는 경우 방지)
        if "용도" in B.columns:
            B = B[B["용도"].str.contains("산업", na=False)]

        B["Period"] = as_period_key(B["날짜"], gran_focus)

        # 업종 상위 N개 선별(전체 사용량 기준)
        top_inds = (
            B.groupby("업종", as_index=False)["사용량"].sum()
             .nlargest(int(max_inds), "사용량")["업종"]
        )
        B = B[B["업종"].isin(top_inds)].copy()

        # ① 히트맵(업종×기간)
        pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
        pivot = pivot[pivot.columns.sort_values()].sort_index()
        Z = pivot.values
        X = pivot.columns.tolist()
        Y = pivot.index.tolist()

        # 라벨(천단위 콤마) 표기
        text = np.vectorize(lambda v: f"{v:,.0f}")(Z)
        zmid = float(np.nanmean(Z)) if np.size(Z) and np.isfinite(Z).any() else None

        heat = go.Figure(data=go.Heatmap(
            z=Z, x=X, y=Y, colorscale="Blues", zmid=zmid,
            colorbar=dict(title="사용량"),
            text=text, texttemplate="%{text}", textfont={"size":11},
            hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(
            template="simple_white", height=560,
            xaxis=dict(title="Period"), yaxis=dict(title="업종"),
            font=dict(family=FONT, size=13), margin=dict(l=70,r=20,t=40,b=40)
        )
        clicked = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                                override_height=560, override_width="100%")

        # 클릭 미지원/미사용 대안: 셀렉터
        sel_ind = None
        sel_period = None
        if clicked:
            c = clicked[0]
            # plotly_events는 좌표가 index/col의 위치를 돌려줌
            # 안전 처리
            try:
                sel_period = X[c["x"]]
                sel_ind = Y[c["y"]]
            except Exception:
                sel_period, sel_ind = None, None

        if not clicked or sel_ind is None or sel_period is None:
            c1, c2 = st.columns(2)
            with c1:
                sel_ind = st.selectbox("업종 선택", Y, index=0 if len(Y) else None)
            with c2:
                sel_period = st.selectbox("기간 선택", X, index=len(X)-1 if len(X) else None)

        if sel_ind and sel_period:
            st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")

            prev_map={"월":12,"분기":4,"반기":2,"연간":1}
            yo = yoy_compare(B[B["업종"]==sel_ind], ["업종","고객명"], "사용량", "Period", prev_map)
            yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)

            # 표 포맷
            yo_sel["사용량"]=yo_sel["사용량"].round(0)
            yo_sel["전년동기"]=yo_sel["전년동기"].round(0)
            yo_sel["증감"]=yo_sel["증감"].round(0)
            yo_sel["YoY(%)"]=yo_sel["YoY(%)"].round(1)

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
                fig_bar.update_layout(
                    template="simple_white", height=520,
                    xaxis=dict(title="고객명", tickangle=-45),
                    yaxis=dict(title="사용량"),
                    font=dict(family=FONT, size=12),
                    margin=dict(l=40,r=20,t=10,b=120)
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("히트맵 셀을 클릭하거나 상단 셀렉터에서 업종/기간을 선택하면 고객 Top-N이 표시돼.")

# ───────────────────────── 사용 파일 확인 ─────────────────────────
with st.expander("🔎 분석에 사용된 원천 파일"):
    if used_overall: st.write(f"A(월별 총괄): **{used_overall}**")
    if used_inds: st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds)>10 else ""))
