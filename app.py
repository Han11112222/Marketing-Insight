# app.py — Fast-first: A는 즉시, B는 탭3에서 지연 로딩
import os, glob, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ───────── Common utils ─────────
def to_num(x):
    if isinstance(x, str): x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    d = pd.to_datetime(dt)
    if gran == "월":   return d.dt.to_period("M").astype(str)
    if gran == "분기": return d.dt.to_period("Q").astype(str)
    if gran == "반기":
        y = d.dt.year.astype(str); h = np.where(d.dt.month<=6, "H1", "H2")
        return y + h
    return d.dt.year.astype(str)

def yoy_compare(df, key_cols, value_col, period_col, gran: str):
    lag_map = {"월":12,"분기":4,"반기":2,"연간":1}
    lag = lag_map.get(gran,12)
    p = df[period_col].astype(str)
    if gran in ["월","분기"]:
        prev = (pd.PeriodIndex(p) - lag).astype(str)
    elif gran=="반기":
        y = p.str[:4].astype(int); h = p.str[-2:].map({"H1":1,"H2":2}).astype(int)
        idx = (y-y.min())*2 + (h-1); prev_idx = idx-2; base=y.min()
        prev = ((prev_idx//2)+base).astype(str) + np.where((prev_idx%2)==0,"H1","H2")
    else:
        prev = (p.astype(int)-1).astype(str)

    cur=df.copy(); cur["_prev"]=prev
    a = cur.groupby(key_cols+[period_col], as_index=False)[value_col].sum()
    b = (cur.rename(columns={period_col:"_prev"})
             .groupby(key_cols+["_prev"], as_index=False)[value_col].sum()
             .rename(columns={value_col:"전년동기"}))
    out = pd.merge(a,b, how="left", left_on=key_cols+[period_col], right_on=key_cols+["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col]-out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs()>1e-9, out["증감"]/out["전년동기"]*100, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def read_parquet_fast(path): return pd.read_parquet(path, engine="pyarrow")

@st.cache_data(ttl=600, show_spinner=False)
def read_csv_any(path):
    for enc in ("utf-8-sig","cp949","euc-kr","utf-8"):
        try: return pd.read_csv(path, encoding=enc, low_memory=False)
        except: pass
    return pd.read_csv(path, encoding_errors="ignore", low_memory=False)

@st.cache_data(ttl=600, show_spinner=False)
def read_excel_any(path_or_buf):
    try: return pd.read_excel(path_or_buf)
    except: return pd.read_excel(path_or_buf, engine="openpyxl")

def find_first(cands):
    for p in cands:
        if os.path.exists(p): return p
    return None

def list_existing(patterns):
    out=[]
    for pat in patterns: out+=glob.glob(pat)
    return sorted(set(out))

# ───────── Sidebar — A만 먼저 ─────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A는 즉시 표시, B는 탭3에서 로딩")

t0 = time.perf_counter()

up_overall = st.sidebar.file_uploader("A) 월별 총괄 (parquet/csv/xlsx)", type=["parquet","csv","xlsx"])
used_overall=None
if up_overall:
    used_overall = up_overall.name
    if used_overall.lower().endswith(".parquet"): overall_raw = read_parquet_fast(up_overall)
    elif used_overall.lower().endswith(".csv"):   overall_raw = pd.read_csv(up_overall, encoding="utf-8-sig")
    else:                                          overall_raw = read_excel_any(up_overall)
else:
    used_overall = find_first([
        "상품별판매량.parquet","상품별판매량.csv","상품별판매량.xlsx",
        "월별총괄.parquet","월별총괄.csv","월별총괄.xlsx",
        "overall.parquet","overall.csv","overall.xlsx"
    ])
    if used_overall:
        if used_overall.lower().endswith(".parquet"): overall_raw = read_parquet_fast(used_overall)
        elif used_overall.lower().endswith(".csv"):   overall_raw = read_csv_any(used_overall)
        else:                                          overall_raw = read_excel_any(used_overall)
        st.sidebar.info(f"A 자동 사용: **{used_overall}**")
    else:
        st.warning("A(월별 총괄) 파일이 필요해요. `상품별판매량.parquet/csv/xlsx` 중 하나를 업로드/저장해 주세요.")
        st.stop()

tA_loaded = time.perf_counter()

# ── A 컬럼 매핑
colsA = overall_raw.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")
def _pickA(keys, default_idx=0):
    for k in keys:
        for c in colsA:
            if k in c: return c
    return colsA[default_idx]

c_date   = st.sidebar.selectbox("날짜", colsA, index=colsA.index(_pickA(["날짜","Date","월"])) if _pickA(["날짜","Date","월"]) in colsA else 0)
c_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 1)
c_indh   = st.sidebar.selectbox("개별난방", colsA, index=colsA.index(_pickA(["개별난방"])) if _pickA(["개별난방"]) in colsA else 2)
c_cenh   = st.sidebar.selectbox("중앙난방", colsA, index=colsA.index(_pickA(["중앙난방"])) if _pickA(["중앙난방"]) in colsA else 3)
c_self   = st.sidebar.selectbox("자가열전용", colsA, index=colsA.index(_pickA(["자가열전용","자가열"])) if _pickA(["자가열전용","자가열"]) in colsA else 4)
c_indusA = st.sidebar.selectbox("산업용 합계", colsA, index=colsA.index(_pickA(["산업용"])) if _pickA(["산업용"]) in colsA else 5)

CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","열병합용1","열병합용2","일반용","일반용(1)","일반용(2)"]
extra_present = [c for c in CAND_EXTRA if c in colsA]
extra_selects={}
if extra_present:
    st.sidebar.markdown("**추가 용도(선택적)**")
    for nm in extra_present:
        extra_selects[nm] = st.sidebar.selectbox(nm, colsA, index=colsA.index(nm))

overall = overall_raw.copy()
overall["날짜"]     = pd.to_datetime(overall[c_date], errors="coerce")
overall["취사용"]   = overall[c_cook].apply(to_num)
overall["개별난방"] = overall[c_indh].apply(to_num)
overall["중앙난방"] = overall[c_cenh].apply(to_num)
overall["자가열전용"]= overall[c_self].apply(to_num)
overall["산업용"]   = overall[c_indusA].apply(to_num)
overall["주택용"]   = overall[["취사용","개별난방","중앙난방","자가열전용"]].sum(axis=1)
for nm,col in extra_selects.items(): overall[nm]=overall[col].apply(to_num)

tA_ready = time.perf_counter()

# ───────── Range & Tabs ─────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")
date_min = overall["날짜"].min()
date_max = overall["날짜"].max()
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

tab0, tab1, tab2 = st.tabs(["🏠 대시보드","📚 집계","🏭 산업용 집중분석"])

# ── 탭0
with tab0:
    st.subheader("연도별 용도 누적 스택")
    A = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
    A["연도"] = A["날짜"].dt.year
    usage_cols = ["주택용","산업용"] + [c for c in CAND_EXTRA if c in A.columns]
    annual = A.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")
    fig0 = go.Figure()
    for col in usage_cols:
        fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
    fig0.update_layout(barmode="stack", template="simple_white", height=420,
                       xaxis=dict(title="Year"), yaxis=dict(title="사용량"),
                       font=dict(family=FONT, size=13),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig0, use_container_width=True, config={"displaylogo": False})
    st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

# ── 탭1
with tab1:
    st.subheader("집계 — 월/분기/반기/연간 (주택용 / 산업용)")
    gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
    A = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
    A["Period"] = as_period_key(A["날짜"], gran)
    sum_tbl = A.groupby("Period", as_index=False)[["주택용","산업용"]].sum().sort_values("Period")
    l,r = st.columns([2,3])
    with l:
        st.dataframe(sum_tbl.style.format({"주택용":"{:,.0f}","산업용":"{:,.0f}"}), use_container_width=True)
    with r:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["주택용"], name="주택용"))
        fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["산업용"], name="산업용"))
        fig.update_layout(barmode="group", template="simple_white", height=360,
                          xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
                          font=dict(family=FONT, size=13))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── 탭2 (여기서만 B 지연 로딩)
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    tB0 = time.perf_counter()

    # 파일 선택 UI (필요할 때만 보여줌)
    st.info("※ 이 탭을 열었을 때만 산업용 상세(B)를 읽습니다.")
    up_indetail = st.file_uploader("B) 산업용 상세 — 여러 개(parquet/csv/xlsx)", type=["parquet","csv","xlsx","xls"], accept_multiple_files=True)
    used_inds=[]

    def read_B_files(files):
        frames=[]
        for f in files:
            used_inds.append(f.name)
            name=f.name.lower()
            if name.endswith(".parquet"): frames.append(pd.read_parquet(f, engine="pyarrow"))
            elif name.endswith(".csv"):   frames.append(pd.read_csv(f, encoding="utf-8-sig"))
            else:                         frames.append(pd.read_excel(f))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if up_indetail:
        indetail_raw = read_B_files(up_indetail)
    else:
        candidate = list_existing(["가정용외_*.parquet","가정용외_*.csv","가정용외_*.xlsx","가정용외_*.xls"])
        if candidate:
            used_inds = [os.path.basename(x) for x in candidate]
            frames=[]
            for p in candidate:
                if p.lower().endswith(".parquet"): frames.append(read_parquet_fast(p))
                elif p.lower().endswith(".csv"):   frames.append(read_csv_any(p))
                else:                               frames.append(read_excel_any(p))
            indetail_raw = pd.concat(frames, ignore_index=True)
            st.caption("B 자동 병합: " + ", ".join(used_inds[:6]) + (" …" if len(used_inds)>6 else ""))
        else:
            st.info("산업용 상세(B) 파일이 없습니다.")
            st.stop()

    tB_read = time.perf_counter()

    colsB = indetail_raw.columns.astype(str).tolist()
    # 최소 매핑 추정
    def _pickB(keys, default=None):
        for k in keys:
            for c in colsB:
                if k in c: return c
        return default if default else (colsB[0] if colsB else None)

    b_date = st.selectbox("날짜(월) 열", colsB, index=(colsB.index(_pickB(["청구년월","사용월","년월"])) if _pickB(["청구년월","사용월","년월"]) in colsB else 0))
    b_use  = st.selectbox("용도 열",   colsB, index=(colsB.index(_pickB(["용도"])) if _pickB(["용도"]) in colsB else 0))
    b_ind  = st.selectbox("업종 열",   colsB, index=(colsB.index(_pickB(["업종"])) if _pickB(["업종"]) in colsB else 0))
    b_cus  = st.selectbox("고객명 열", colsB, index=(colsB.index(_pickB(["고객","고객명","거래처"])) if _pickB(["고객","고객명","거래처"]) in colsB else 0))
    b_amt  = st.selectbox("사용량 열", colsB, index=(colsB.index(_pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"])) if _pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"]) in colsB else 0))

    def parse_month(x):
        s=str(x)
        for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
            try: return pd.to_datetime(s, format=fmt).replace(day=1)
            except: pass
        return pd.to_datetime(s, errors="coerce")

    B = indetail_raw.rename(columns={b_date:"__dt__", b_use:"__use__", b_ind:"__ind__", b_cus:"__cus__", b_amt:"__amt__"})
    B["날짜"] = pd.to_datetime(B["__dt__"].apply(parse_month), errors="coerce")
    B["용도"] = B["__use__"].astype(str).str.strip()
    B["업종"] = B["__ind__"].astype(str).str.strip()
    B["고객명"]= B["__cus__"].astype(str).str.strip()
    B["사용량"]= pd.to_numeric(B["__amt__"].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
    B = B[["날짜","용도","업종","고객명","사용량"]].dropna(subset=["날짜"])

    # 산업용만 필터
    B = B[B["용도"].str.contains("산업", na=False)]
    if B.empty:
        st.info("산업용 행이 없습니다.")
        st.stop()

    tB_ready = time.perf_counter()

    # 기간 선택
    d1_B, d2_B = st.date_input("B 기간", [pd.to_datetime(B["날짜"].min()), pd.to_datetime(B["날짜"].max())], key="b_range")
    B = B[(B["날짜"]>=pd.to_datetime(d1_B)) & (B["날짜"]<=pd.to_datetime(d2_B))].copy()

    gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
    B["Period"] = as_period_key(B["날짜"], gran_focus)

    # 피벗 & 히트맵
    pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
    pivot = pivot[pivot.columns.sort_values()].sort_index()
    Z = pivot.values; X = pivot.columns.tolist(); Y = pivot.index.tolist()
    zmid = float(np.nanmean(Z)) if np.isfinite(Z).all() else None

    def fmt_k(v):
        try:
            v=float(v)
            if v>=1_000_000_000: return f"{v/1_000_000_000:.1f}B"
            if v>=1_000_000:     return f"{v/1_000_000:.1f}M"
            if v>=1_000:         return f"{v/1_000:.1f}k"
            return f"{v:.0f}"
        except: return ""
    text = np.vectorize(fmt_k)(Z)

    heat = go.Figure(data=go.Heatmap(
        z=Z, x=X, y=Y, colorscale="Blues", zmid=zmid, colorbar=dict(title="사용량"),
        text=text, texttemplate="%{text}", textfont={"size":11},
        hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
    ))
    heat.update_layout(template="simple_white", height=560,
                       xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                       font=dict(family=FONT, size=13), margin=dict(l=70,r=20,t=40,b=40))

    if HAS_PLOTLY_EVENTS:
        clicked = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                                override_height=560, override_width="100%")
    else:
        st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
        st.caption("※ 셀 클릭 기능을 사용하려면 requirements.txt에 `streamlit-plotly-events` 추가")

        clicked = []

    if clicked:
        sel_period = str(clicked[0].get("x"))
        sel_ind    = str(clicked[0].get("y"))
        st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")
        yo = yoy_compare(B[B["업종"]==sel_ind], ["업종","고객명"], "사용량", "Period", gran_focus)
        yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)
        yo_sel["사용량"]=yo_sel["사용량"].round(0); yo_sel["전년동기"]=yo_sel["전년동기"].round(0)
        yo_sel["증감"]=yo_sel["증감"].round(0); yo_sel["YoY(%)"]=yo_sel["YoY(%)"].round(1)
        top_n = st.slider("상위 N", 5, 100, 20, step=5)
        view = yo_sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

        c1,c2 = st.columns([1.4,1.6])
        with c1:
            st.dataframe(view.style.format({"사용량":"{:,.0f}","전년동기":"{:,.0f}","증감":"{:+,.0f}","YoY(%)":"{:+,.1f}"}),
                         use_container_width=True, height=520)
            st.download_button("⬇️ 고객리스트 CSV",
                               data=view.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{sel_ind}_{sel_period}_top{top_n}.csv",
                               mime="text/csv")
        with c2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=view["고객명"], y=view["사용량"], name="사용량",
                                     text=[f"{v:,.0f}" for v in view["사용량"]], textposition="auto"))
            fig_bar.update_layout(template="simple_white", height=520,
                                  xaxis=dict(title="고객명", tickangle=-45),
                                  yaxis=dict(title="사용량"),
                                  font=dict(family=FONT, size=12), margin=dict(l=40,r=20,t=10,b=120))
            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

    # ── 타임라인 표시
    tB_done = time.perf_counter()
    st.info(f"⏱ A 읽기 {tA_loaded - t0:.2f}s · A 가공 {tA_ready - tA_loaded:.2f}s · "
            f"B 읽기 {tB_read - tB0:.2f}s · B 가공 {tB_ready - tB_read:.2f}s · "
            f"B 전체 {tB_done - tB0:.2f}s")

# ── 사용 파일
with st.expander("🔎 분석에 사용된 원천 파일"):
    if used_overall: st.write(f"A(월별 총괄): **{used_overall}**")
