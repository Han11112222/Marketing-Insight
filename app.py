# app.py — Gas Sales Analytics (Repo Auto-Load Edition)
# 월/분기/반기/연간 집계 + 산업용 업종 히트맵(클릭→고객리스트/전년대비)
# - 업로드가 없으면 리포지토리의 기본 파일을 자동으로 읽어옴:
#   A) 상품별판매량.xlsx (또는 월별총괄.xlsx)
#   B) 가정용외_*.csv / 가정용외_*.xlsx (여러 개 병합)

import os, io, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
PLOT_FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")

# ─────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────
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
        y = d.dt.year
        h = np.where(d.dt.month <= 6, "H1", "H2")
        return (y.astype(str) + h)
    else:
        return d.dt.year.astype(str)

def yoy_compare(df, key_cols, value_col, period_col, prev_map):
    gran = st.session_state.get("granularity", "월")
    lag_n = prev_map.get(gran, 12)
    p = df[period_col].astype(str)

    if gran in ["월", "분기"]:
        pp = pd.PeriodIndex(p)
        prev = (pp - lag_n).astype(str)
    elif gran == "반기":
        y = p.str.slice(0,4).astype(int)
        h = p.str[-2:].map({"H1":1, "H2":2}).astype(int)
        idx = (y - y.min())*2 + (h-1)
        prev_idx = idx - 2
        base_y = y.min()
        prev_y = (prev_idx // 2) + base_y
        prev_h = np.where((prev_idx % 2)==0, "H1", "H2")
        prev = (prev_y.astype(str) + prev_h)
    else:
        y = p.astype(int)
        prev = (y - 1).astype(str)

    cur_df = df.copy()
    cur_df["_prev_key"] = prev
    cur_agg = cur_df.groupby(key_cols + [period_col], as_index=False)[value_col].sum()
    prev_agg = cur_df.rename(columns={period_col: "_prev_key"}) \
                     .groupby(key_cols + ["_prev_key"], as_index=False)[value_col].sum() \
                     .rename(columns={value_col: "전년동기"})
    out = pd.merge(cur_agg, prev_agg, how="left",
                   left_on=key_cols + [period_col],
                   right_on=key_cols + ["_prev_key"])
    out.drop(columns=["_prev_key"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs()>1e-9, out["증감"]/out["전년동기"]*100, np.nan)
    return out

# ─────────────────────────────────────────
# 리포지토리 파일 자동탐색 로더
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _read_excel_any(path_or_buf):
    try:
        return pd.read_excel(path_or_buf)
    except:
        return pd.read_excel(path_or_buf, engine="openpyxl")

@st.cache_data(show_spinner=False)
def _read_csv_any(path):
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding_errors="ignore")

def find_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def list_existing(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(list(set(files)))

# ─────────────────────────────────────────
# A) 월별 총괄 입력 — 업로드 OR 리포 자동로드
# ─────────────────────────────────────────
st.sidebar.header("① 데이터 업로드")
st.sidebar.caption("A: 월별 총괄(주택/산업 합산), B: 산업용 상세(고객/업종)")

file_overall_up = st.sidebar.file_uploader("A) 월별 총괄 엑셀(.xlsx)", type=["xlsx"], key="overall")

used_overall_path = None
if file_overall_up:
    overall = _read_excel_any(file_overall_up)
    used_overall_path = file_overall_up.name
else:
    # 리포지토리 기본 파일 자동 탐색
    overall_candidates = [
        "상품별판매량.xlsx",     # 한엽님 저장명
        "월별총괄.xlsx",        # 예비명
        "overall.xlsx"
    ]
    used_overall_path = find_first_existing(overall_candidates)
    if used_overall_path:
        overall = _read_excel_any(used_overall_path)
        st.info(f"리포지토리 기본 파일 자동 사용: **{used_overall_path}**")
    else:
        st.warning("좌측에서 A(월별 총괄) 파일을 업로드하거나, 리포지토리에 `상품별판매량.xlsx` 를 넣어줘.")
        st.stop()

colsA = overall.columns.astype(str).tolist()
st.sidebar.header("② A(월별 총괄) 컬럼 매핑")
def _pickA(cands, default_idx=0):
    for k in cands:
        for c in colsA:
            if k in str(c):
                return c
    return colsA[default_idx]

col_date_A = st.sidebar.selectbox("날짜", colsA, index=colsA.index(_pickA(["날짜","date","Date","월"])) if _pickA(["날짜","date","Date","월"]) in colsA else 0)
col_cook   = st.sidebar.selectbox("취사용", colsA, index=colsA.index(_pickA(["취사용"])) if _pickA(["취사용"]) in colsA else 1)
col_indh   = st.sidebar.selectbox("개별난방", colsA, index=colsA.index(_pickA(["개별난방"])) if _pickA(["개별난방"]) in colsA else 2)
col_cenh   = st.sidebar.selectbox("중앙난방", colsA, index=colsA.index(_pickA(["중앙난방"])) if _pickA(["중앙난방"]) in colsA else 3)
col_self   = st.sidebar.selectbox("자가열전용", colsA, index=colsA.index(_pickA(["자가열전용","자가열"])) if _pickA(["자가열전용","자가열"]) in colsA else 4)
col_indusA = st.sidebar.selectbox("산업용 합계", colsA, index=colsA.index(_pickA(["산업용"])) if _pickA(["산업용"]) in colsA else 5)

overall_df = overall.copy()
overall_df["날짜"] = pd.to_datetime(overall_df[col_date_A], errors="coerce")
overall_df["취사용"] = overall_df[col_cook].apply(to_num)
overall_df["개별난방"] = overall_df[col_indh].apply(to_num)
overall_df["중앙난방"] = overall_df[col_cenh].apply(to_num)
overall_df["자가열전용"] = overall_df[col_self].apply(to_num)
overall_df["산업용"] = overall_df[col_indusA].apply(to_num)
overall_df["주택용"] = overall_df[["취사용","개별난방","중앙난방","자가열전용"]].sum(axis=1)

# ─────────────────────────────────────────
# B) 산업용 상세 — 업로드 OR 리포 자동 병합
# ─────────────────────────────────────────
files_industrial_up = st.sidebar.file_uploader(
    "B) 산업용 상세 파일(여러 개 선택 가능) — CSV/XLSX 혼용 가능",
    type=["csv","xlsx","xls"], accept_multiple_files=True, key="indetail_multi"
)

used_industry_files = []
if files_industrial_up:
    frames = []
    for f in files_industrial_up:
        used_industry_files.append(f.name)
        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(f, encoding="utf-8-sig")
            else:
                df = pd.read_excel(f)
        except Exception:
            f.seek(0)
            df = pd.read_excel(f, engine="openpyxl")
        df["__file__"] = f.name
        frames.append(df)
    indetail = pd.concat(frames, ignore_index=True)
else:
    # 리포지토리 자동 수집 (가정용외_*.csv / *.xlsx)
    patterns = ["가정용외_*.csv", "가정용외_*.xlsx", "가정용외_*.xls"]
    candidates = list_existing(patterns)
    if candidates:
        frames = []
        for p in candidates:
            used_industry_files.append(os.path.basename(p))
            if p.lower().endswith(".csv"):
                df = _read_csv_any(p)
            else:
                df = _read_excel_any(p)
            df["__file__"] = os.path.basename(p)
            frames.append(df)
        indetail = pd.concat(frames, ignore_index=True)
        st.info(f"리포지토리 산업용 상세 자동 병합: **{', '.join(used_industry_files)}**")
    else:
        st.warning("좌측에서 B(산업용 상세) 파일을 업로드하거나, 리포지토리에 `가정용외_*.csv` 파일들을 넣어줘.")
        st.stop()

colsB = indetail.columns.astype(str).tolist()
st.sidebar.header("③ B(산업용 상세) 컬럼 매핑")
def _pickB(cands, default=None):
    for k in cands:
        for c in colsB:
            if k in c:
                return c
    return default if default is not None else colsB[0]

col_date_B = st.sidebar.selectbox("날짜(월 기준)", colsB,
    index=colsB.index(_pickB(["청구년월","사용월","년월","월"])) if _pickB(["청구년월","사용월","년월","월"]) in colsB else 0)
col_use_tp = st.sidebar.selectbox("용도(영업용/업무용/산업용…)", colsB,
    index=colsB.index(_pickB(["용도"])) if _pickB(["용도"]) in colsB else 0)
col_indus  = st.sidebar.selectbox("업종", colsB,
    index=colsB.index(_pickB(["업종"])) if _pickB(["업종"]) in colsB else 0)
col_cust   = st.sidebar.selectbox("고객명", colsB,
    index=colsB.index(_pickB(["고객","고객명","거래처","업체"])) if _pickB(["고객","고객명","거래처","업체"]) in colsB else 0)
col_usage  = st.sidebar.selectbox("사용량(예: 사용량(m3사용량))", colsB,
    index=colsB.index(_pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"])) if _pickB(["사용량(m3","m3사용량","사용량","수량","NM3","Nm3","MJ"]) in colsB else 0)

def _parse_month(s):
    s = str(s)
    for fmt in ["%Y-%m", "%Y/%m", "%Y%m", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m"]:
        try:
            d = pd.to_datetime(s, format=fmt)
            return pd.Timestamp(d.year, d.month, 1)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

indetail_df = indetail.copy()
indetail_df["날짜"] = pd.to_datetime(indetail_df[col_date_B].apply(_parse_month), errors="coerce")
indetail_df["용도"] = indetail_df[col_use_tp].astype(str).str.strip()
indetail_df["업종"] = indetail_df[col_indus].astype(str).str.strip()
indetail_df["고객명"] = indetail_df[col_cust].astype(str).str.strip()
indetail_df["사용량"] = pd.to_numeric(
    indetail_df[col_usage].astype(str).str.replace(",","").str.replace(" ", ""), errors="coerce"
).fillna(0)

# ─────────────────────────────────────────
# 분석 옵션
# ─────────────────────────────────────────
st.sidebar.header("④ 분석 옵션")
gran = st.sidebar.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
unit = st.sidebar.selectbox("표시 단위", ["MJ","Nm³"], index=0)

date_min = min(overall_df["날짜"].min(), indetail_df["날짜"].min())
date_max = max(overall_df["날짜"].max(), indetail_df["날짜"].max())
d1, d2 = st.sidebar.date_input("기간", [pd.to_datetime(date_min), pd.to_datetime(date_max)])

UNIT_FACTOR = 1.0      # 필요 시 MJ↔Nm³ 환산 사용
display_unit = unit

# 사용한 파일 안내
with st.expander("🔎 이번 분석에 사용된 원천 파일"):
    if used_overall_path:
        st.write(f"A(월별 총괄): **{used_overall_path}**")
    if used_industry_files:
        st.write("B(산업용 상세):")
        st.write(", ".join(used_industry_files[:8]) + (" …" if len(used_industry_files)>8 else ""))

# ─────────────────────────────────────────
# ① 월/분기/반기/연간 집계
# ─────────────────────────────────────────
st.subheader("① 월/분기/반기/연간 집계 (주택용 / 산업용)")
maskA = (overall_df["날짜"] >= pd.to_datetime(d1)) & (overall_df["날짜"] <= pd.to_datetime(d2))
A = overall_df.loc[maskA].copy()
A["Period"] = as_period_key(A["날짜"], gran)

sum_tbl = A.groupby("Period", as_index=False)[["주택용","산업용"]].sum().sort_values("Period")
sum_tbl_disp = sum_tbl.copy()
if display_unit == "Nm³":
    sum_tbl_disp[["주택용","산업용"]] = sum_tbl_disp[["주택용","산업용"]] / UNIT_FACTOR

col1, col2 = st.columns([2,3])
with col1:
    st.dataframe(sum_tbl_disp.style.format({"주택용":"{:,.0f}","산업용":"{:,.0f}"}), use_container_width=True)
with col2:
    fig_sum = go.Figure()
    fig_sum.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["주택용"], name="주택용"))
    fig_sum.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl["산업용"], name="산업용"))
    fig_sum.update_layout(barmode="group", template="simple_white",
                          xaxis=dict(title="Period"),
                          yaxis=dict(title=f"사용량 ({display_unit})"),
                          height=360, font=dict(family=PLOT_FONT, size=13))
    st.plotly_chart(fig_sum, use_container_width=True, config={"displaylogo": False})

st.divider()

# ─────────────────────────────────────────
# ② 산업용 — 업종 히트맵 & 클릭 → 고객 리스트
# ─────────────────────────────────────────
st.subheader("② 산업용 — 업종 히트맵  →  클릭 시 고객리스트(상위/전년대비)")

maskB = (indetail_df["날짜"] >= pd.to_datetime(d1)) & (indetail_df["날짜"] <= pd.to_datetime(d2))
B = indetail_df.loc[maskB].copy()
B["Period"] = as_period_key(B["날짜"], gran)

pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
pivot = pivot.sort_index(axis=0)
pivot = pivot[pivot.columns.sort_values()]
Z = pivot.values
X = pivot.columns.tolist()
Y = pivot.index.tolist()
zmid = float(np.nanmean(Z)) if np.isfinite(Z).all() else None

heat = go.Figure(data=go.Heatmap(
    z=Z, x=X, y=Y, colorscale="Blues", zmid=zmid, colorbar=dict(title=display_unit),
    hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f} "+display_unit+"<extra></extra>"
))
heat.update_layout(template="simple_white", height=520,
                   xaxis=dict(title="Period", tickangle=0),
                   yaxis=dict(title="업종"),
                   font=dict(family=PLOT_FONT, size=13),
                   margin=dict(l=60,r=20,t=40,b=40))

clicked = plotly_events(heat, click_event=True, hover_event=False,
                        select_event=False, override_height=520, override_width="100%")

st.caption("힌트: 히트맵 셀을 클릭하면 오른쪽 표에 해당 업종·기간의 고객 상위/전년대비가 표시돼.")

colL, colR = st.columns([1.0, 1.4])
with colR:
    if clicked:
        c = clicked[0]
        sel_period = X[c["x"]]
        sel_industry = Y[c["y"]]
        st.markdown(f"**선택 업종:** `{sel_industry}` · **선택 기간:** `{sel_period}`")

        prev_map = {"월":12, "분기":4, "반기":2, "연간":1}
        yo = yoy_compare(B[B["업종"]==sel_industry],
                         key_cols=["업종","고객명"],
                         value_col="사용량",
                         period_col="Period",
                         prev_map=prev_map)
        yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)

        yo_sel["사용량"]   = yo_sel["사용량"].round(0)
        yo_sel["전년동기"] = yo_sel["전년동기"].round(0)
        yo_sel["증감"]     = yo_sel["증감"].round(0)
        yo_sel["YoY(%)"]   = yo_sel["YoY(%)"].round(1)

        top_n = st.selectbox("상위 N", [10,20,50,100], index=1)
        view = yo_sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

        st.dataframe(
            view.style.format({"사용량":"{:,.0f}","전년동기":"{:,.0f}","증감":"{:+,.0f}","YoY(%)":"{:+,.1f}"}),
            use_container_width=True, height=520
        )
        csv = view.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 고객리스트 CSV 다운로드", data=csv,
                           file_name=f"{sel_industry}_{sel_period}_top{top_n}.csv",
                           mime="text/csv")
    else:
        st.info("히트맵에서 업종·기간 셀을 클릭하면 고객 리스트가 여기에 표시돼.")
