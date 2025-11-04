# app.py — Gas Sales Analytics (Landing + Aggregations + Industrial Focus)
# - 탭3: [산업용 집중분석] 업종×기간 히트맵 → 셀 클릭: 고객 Top-N / YoY / 다운로드

import os, glob, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# ───────────── 공통 유틸 ─────────────
def to_num(x):
    if isinstance(x, str): x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def as_period_key(dt: pd.Series, gran: str) -> pd.Series:
    d = pd.to_datetime(dt)
    if gran == "월":
        return d.dt.to_period("M").astype(str)
    elif gran == "분기":
        return d.dt.to_period("Q").astype(str)
    elif gran == "반기":
        y = d.dt.year
        h = np.where(d.dt.month<=6, "H1", "H2")
        return (y.astype(str)+h)
    else:
        return d.dt.year.astype(str)

def yoy_compare(df, key_cols, value_col, period_col, prev_map):
    gran = st.session_state.get("granularity","월")
    lag = prev_map.get(gran,12)
    p = df[period_col].astype(str)
    if gran in ["월","분기"]:
        prev = (pd.PeriodIndex(p) - lag).astype(str)
    elif gran=="반기":
        y = p.str[:4].astype(int)
        h = p.str[-2:].map({"H1":1,"H2":2}).astype(int)
        idx = (y-y.min())*2 + (h-1)
        prev_idx = idx-2
        base = y.min()
        prev = ( (prev_idx//2)+base ).astype(str) + np.where((prev_idx%2)==0,"H1","H2")
    else:
        prev = (p.astype(int)-1).astype(str)
    cur = df.copy(); cur["_prev"] = prev
    a = cur.groupby(key_cols+[period_col], as_index=False)[value_col].sum()
    b = cur.rename(columns={period_col:"_prev"}).groupby(key_cols+["_prev"], as_index=False)[value_col].sum().rename(columns={value_col:"전년동기"})
    out = pd.merge(a,b, how="left", left_on=key_cols+[period_col], right_on=key_cols+["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col]-out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs()>1e-9, out["증감"]/out["전년동기"]*100, np.nan)
    return out

@st.cache_data(show_spinner=False)
def read_excel_any(path_or_buf):
    try: return pd.read_excel(path_or_buf)
    except: return pd.read_excel(path_or_buf, engine="openpyxl")

@st.cache_data(show_spinner=False)
def read_csv_any(path_or_buf):
    for enc in ["cp949","euc-kr","utf-8-sig","utf-8"]:
        try: return pd.read_csv(path_or_buf, encoding=enc)
        except Exception: continue
    return pd.read_csv(path_or_buf, encoding_errors="ignore")

@st.cache_data(show_spinner=False)
def read_parquet_any(path_or_buf):
    # 업로드 객체(file-like) 또는 경로 모두 지원
    return pd.read_parquet(path_or_buf, engine="pyarrow")

def find_first(cands):
    for p in cands:
        if os.path.exists(p): return p
    return None

def list_existing(patterns):
    out=[]
    for pat in patterns: out+=glob.glob(pat)
    return sorted(set(out))

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        s = str(s).strip()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[\(\)\[\]{}㎥/NnMmJj]+", "", s)  # 단위/괄호류 제거
        return s.lower()
    m = {c: norm(c) for c in df.columns}
    return df.rename(columns=m)

def pick_col(cols, *keys, default=None):
    for k in keys:
        for c in cols:
            if k in c:
                return c
    return default

# ───────────── 데이터 입력(사이드바) ─────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종)")

# A) 월별 총괄
up_overall = st.sidebar.file_uploader("A) 월별 총괄 엑셀(.xlsx)", type=["xlsx"])
if up_overall:
    overall_raw = read_excel_any(up_overall)
    used_overall = up_overall.name
else:
    used_overall = find_first(["상품별판매량.xlsx","월별총괄.xlsx","overall.xlsx"])
    if used_overall:
        overall_raw = read_excel_any(used_overall)
        st.sidebar.info(f"A 자동 사용: **{used_overall}**")
    else:
        st.info("A(월별 총괄)를 업로드하거나 `상품별판매량.xlsx`를 저장소에 넣어줘.")
        st.stop()

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

CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","일반용"]
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

# B) 산업용 상세 — parquet 우선, csv/xlsx도 허용
st.sidebar.header("③ B(산업용 상세) 업로드/자동탐지")
up_indetail = st.sidebar.file_uploader(
    "B) 산업용 상세 — Parquet/CSV/XLSX 여러 개 업로드 가능",
    type=["parquet","csv","xlsx","xls"], accept_multiple_files=True
)
used_inds = []
if up_indetail:
    frames=[]
    for f in up_indetail:
        used_inds.append(f.name)
        name=f.name.lower()
        if   name.endswith(".parquet"): df = read_parquet_any(f)
        elif name.endswith(".csv"):     df = read_csv_any(f)
        else:                           df = read_excel_any(f)
        frames.append(df)
    indetail_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
else:
    # 레포 루트의 파일 자동탐지 (parquet 우선)
    pats_pq  = ["가정용외_*.parquet", "parquet_out/가정용외_*.parquet"]
    pats_csv = ["가정용외_*.csv"]
    pats_xls = ["가정용외_*.xlsx","가정용외_*.xls"]
    files = list_existing(pats_pq) or list_existing(pats_csv) or list_existing(pats_xls)
    if files:
        used_inds = [os.path.basename(p) for p in files]
        frames=[]
        for p in files:
            if p.lower().endswith(".parquet"): df = read_parquet_any(p)
            elif p.lower().endswith(".csv"):   df = read_csv_any(p)
            else:                               df = read_excel_any(p)
            frames.append(df)
        indetail_raw = pd.concat(frames, ignore_index=True)
        st.sidebar.info("B 자동 병합: " + ", ".join(used_inds[:6]) + (" …" if len(used_inds)>6 else ""))
    else:
        indetail_raw = pd.DataFrame()

# ── B 컬럼 자동 매핑(파일마다 이름 차이 흡수) ──
def build_indetail(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])
    Bn = normalize_cols(df_raw)
    cols = list(Bn.columns)

    col_date = pick_col(cols, "청구년월","사용월","년월","청구월","월","년월일","일자", default=None)
    col_use  = pick_col(cols, "용도", default=None)
    col_ind  = pick_col(cols, "업종","업종분류","표준산업분류", default=None)
    col_cus  = pick_col(cols, "고객명","고객","거래처","수요처", default=None)
    col_amt  = pick_col(cols, "사용량m3","m3사용량","사용량","수량","nm3","실사용", "mj", default=None)

    # 역-매핑(원래 컬럼명 찾기)
    def get_raw(colnorm):
        for c in df_raw.columns:
            cc = re.sub(r"\s+","",str(c)).lower()
            cc = re.sub(r"[\(\)\[\]{}㎥/NnMmJj]+","",cc)
            if colnorm == cc:
                return c
        return None

    need = [col_date,col_use,col_ind,col_cus,col_amt]
    if any(v is None for v in need):
        return pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

    out = pd.DataFrame({
        "날짜": pd.to_datetime(
            df_raw[get_raw(col_date)].astype(str).str.replace(r"[^\d\-\/\.]","", regex=True),
            errors="coerce"
        ).dt.to_period("M").dt.to_timestamp(),
        "용도": df_raw[get_raw(col_use)].astype(str).str.strip(),
        "업종": df_raw[get_raw(col_ind)].astype(str).str.strip(),
        "고객명": df_raw[get_raw(col_cus)].astype(str).str.strip(),
        "사용량": pd.to_numeric(df_raw[get_raw(col_amt)].astype(str).str.replace(",",""), errors="coerce").fillna(0)
    })
    return out

indetail = build_indetail(indetail_raw)

# ───────────── 범위/단위 ─────────────
st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")

def _safe_minmax(series: pd.Series):
    """NaT/결측 제거 후 (min, max) 반환. 모두 NaT면 오늘 기준 최근 24개월로."""
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        # 기본: 오늘 기준 최근 24개월
        today = pd.Timestamp.today().normalize()
        return (today - pd.DateOffset(months=24)).replace(day=1), today
    return s.min(), s.max()

# A, B 각각에서 안전한 min/max 구한 뒤 전체 min/max
a_min, a_max = _safe_minmax(overall["날짜"])
if 'indetail' in locals() and len(indetail) > 0:
    b_min, b_max = _safe_minmax(indetail["날짜"])
    date_min = min(a_min, b_min)
    date_max = max(a_max, b_max)
else:
    date_min, date_max = a_min, a_max

# min > max가 되는 상황 방지 (같으면 max에 +1일 버퍼)
if pd.isna(date_min) or pd.isna(date_max) or date_min > date_max:
    today = pd.Timestamp.today().normalize()
    date_min = (today - pd.DateOffset(months=24)).replace(day=1)
    date_max = today

d1, d2 = st.sidebar.date_input(
    "기간",
    [pd.to_datetime(date_min), pd.to_datetime(date_max)]
)

# ── 탭2: 산업용 집중분석 ──
with tab2:
    st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N")
    if len(indetail)==0:
        st.info("산업용 상세 파일(B)이 없어 히트맵을 표시할 수 없어.")
        st.stop()

    # 산업용만 필터 (파일에 다른 용도가 섞여 있는 경우)
    B = indetail[(indetail["날짜"]>=pd.to_datetime(d1)) & (indetail["날짜"]<=pd.to_datetime(d2))].copy()
    if "용도" in B.columns:
        B = B[B["용도"].astype(str).str.contains("산업", na=False)]

    gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
    B["Period"] = as_period_key(B["날짜"], gran_focus)

    # ① 히트맵(업종×기간)
    pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
    if pivot.empty:
        st.warning("표시할 데이터가 없어.")
        st.stop()
    pivot = pivot[pivot.columns.sort_values()].sort_index()

    Z = pivot.values
    X = pivot.columns.tolist()
    Y = pivot.index.tolist()
    text = np.vectorize(lambda v: f"{v:,.0f}")(Z)  # 셀 라벨(사용량)

    heat = go.Figure(
        data=go.Heatmap(
            z=Z, x=X, y=Y,
            colorscale="Blues",
            colorbar=dict(title="사용량"),
            text=text, texttemplate="%{text}", textfont={"size":12},
            hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
        )
    )
    heat.update_layout(template="simple_white", height=560,
                       xaxis=dict(title="Period"),
                       yaxis=dict(title="업종"),
                       font=dict(family=FONT, size=13),
                       margin=dict(l=80,r=20,t=40,b=40))
    clicked = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                            override_height=560, override_width="100%")

    # ② 클릭 후: 고객 Top-N + 막대그래프
    if clicked:
        c = clicked[0]
        sel_period = X[c["x"]]
        sel_ind    = Y[c["y"]]
        st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")

        prev_map={"월":12,"분기":4,"반기":2,"연간":1}
        yo = yoy_compare(B[B["업종"]==sel_ind], ["업종","고객명"], "사용량", "Period", prev_map)
        yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)

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
            fig_bar.update_layout(template="simple_white", height=520,
                                  xaxis=dict(title="고객명", tickangle=-45),
                                  yaxis=dict(title="사용량"),
                                  font=dict(family=FONT, size=12),
                                  margin=dict(l=40,r=20,t=10,b=120))
            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("히트맵 셀을 클릭하면 아래에 고객 Top-N과 막대그래프가 표시돼.")

# ───────────── 사용 파일 확인 ─────────────
with st.expander("🔎 분석에 사용된 원천 파일"):
    if 'used_overall' in locals() and used_overall:
        st.write(f"A(월별 총괄): **{used_overall}**")
    if used_inds:
        st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds)>10 else ""))
