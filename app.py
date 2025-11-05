# app.py — Gas Sales Analytics (Landing + Aggregations + Industrial Focus)
# - 탭3: [산업용 집중분석] 업종×기간 히트맵 → 셀 클릭: 고객 Top-N / YoY / 다운로드
# - 파일: Parquet 우선(업로드 또는 저장소 자동탐색)
# - 런타임 에러는 화면에 바로 표시

import os, glob, io, contextlib, traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# (선택) plotly-events 없으면 대체 루틴 사용
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False
    def plotly_events(fig, **kwargs):
        st.info("`streamlit-plotly-events`가 없어 셀 클릭 기능을 비활성화합니다. 상단 필터를 사용하세요.")
        return []

# -------------------- 기본 설정 --------------------
st.set_page_config(page_title="도시가스 판매량 분석", layout="wide")
FONT = "Noto Sans KR, Pretendard, Arial, sans-serif"

# 랜딩 스택에 보조로 포함할 수 있는 후보 열(있으면 자동 포함)
CAND_EXTRA = ["수송용","업무용","연료전지용","열전용설비용","열병합용","열병합용1","열병합용2","일반용","일반용(1)","일반용(2)"]

# -------------------- 에러 보여주기 래퍼 --------------------
@contextlib.contextmanager
def show_errors():
    try:
        yield
    except Exception as e:
        st.error("❌ 앱 실행 중 예외가 발생했습니다.")
        st.exception(e)
        tb = traceback.format_exc()
        with st.expander("자세한 스택트레이스 열기"):
            st.code(tb, language="python")
        st.stop()

# -------------------- 캐시 I/O --------------------
@st.cache_data(show_spinner=False)
def read_parquet_any(buf_or_path):
    return pd.read_parquet(buf_or_path)

@st.cache_data(show_spinner=False)
def list_existing(patterns):
    out = []
    for pat in patterns:
        out += glob.glob(pat)
    return sorted(set(out))

# -------------------- 공통 유틸 --------------------
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "").replace(" ", "")
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

    if gran in ["월","분기"]:
        prev = (pd.PeriodIndex(p) - lag).astype(str)
    elif gran == "반기":
        y = p.str[:4].astype(int)
        h = p.str[-2:].map({"H1":1,"H2":2}).astype(int)
        idx = (y - y.min())*2 + (h-1)
        prev_idx = idx - 2
        base = y.min()
        prev = ((prev_idx//2)+base).astype(str) + np.where((prev_idx%2)==0, "H1","H2")
    else:
        prev = (p.astype(int) - 1).astype(str)

    cur = df.copy()
    cur["_prev"] = prev

    a = cur.groupby(key_cols + [period_col], as_index=False)[value_col].sum()
    b = (
        cur.rename(columns={period_col:"_prev"})
          .groupby(key_cols + ["_prev"], as_index=False)[value_col]
          .sum()
          .rename(columns={value_col:"전년동기"})
    )
    out = pd.merge(a, b, how="left",
                   left_on=key_cols+[period_col],
                   right_on=key_cols+["_prev"])
    out.drop(columns=["_prev"], inplace=True, errors="ignore")
    out["증감"] = out[value_col] - out["전년동기"]
    out["YoY(%)"] = np.where(out["전년동기"].abs()>1e-9, out["증감"]/out["전년동기"]*100, np.nan)
    return out

def pick_best(cols, keys, default=None):
    cols = [str(c) for c in cols]
    for k in keys:
        cand = [c for c in cols if k in c]
        if cand:
            return cand[0]
    return default if default is not None else (cols[0] if cols else None)

# ================== 본문 ==================
with show_errors():

    # ---------- 사이드바: 데이터 입력 ----------
    st.sidebar.header("① 데이터 업로드")
    st.sidebar.caption("A: 월별 총괄(주택/산업 합산, Parquet) · B: 산업용 상세(고객/업종, Parquet)")

    # A) 업로드(멀티) 또는 저장소 자동탐색
    up_overall_files = st.sidebar.file_uploader("A) 월별 총괄 — Parquet (여러 개 선택 가능)", type=["parquet"], accept_multiple_files=True)
    overall_frames = []
    used_overall = []

    if up_overall_files:
        for f in up_overall_files:
            df = read_parquet_any(f)
            overall_frames.append(df)
            used_overall.append(f.name)
    else:
        # 저장소 자동탐색 (가정용외_*.parquet or 유사 패턴 + 상품별판매량은 제외)
        pats = ["*.parquet"]
        cand = [p for p in list_existing(pats) if "상품별판매량" not in os.path.basename(p)]
        # 기간조각이 여러 개라면 모두 로드 후 concat
        for p in cand:
            try:
                df = read_parquet_any(p)
                overall_frames.append(df)
                used_overall.append(os.path.basename(p))
            except Exception:
                pass

    overall_raw = pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame()

    # B) 업로드(멀티) 또는 저장소 자동탐색(상품별판매량.parquet 우선)
    up_indetail = st.sidebar.file_uploader("B) 산업용 상세 — Parquet (여러 개 선택 가능)", type=["parquet"], accept_multiple_files=True)
    indetail_frames, used_inds = [], []

    if up_indetail:
        for f in up_indetail:
            df = read_parquet_any(f)
            indetail_frames.append(df)
            used_inds.append(f.name)
    else:
        patsB = ["*상품별판매량*.parquet", "*산업용*.parquet", "*가정용외_산업*.parquet"]
        candB = list_existing(patsB)
        for p in candB:
            try:
                df = read_parquet_any(p)
                indetail_frames.append(df)
                used_inds.append(os.path.basename(p))
            except Exception:
                pass

    indetail_raw = pd.concat(indetail_frames, ignore_index=True) if indetail_frames else pd.DataFrame()

    # ---------- A 컬럼 매핑 ----------
    st.sidebar.header("② A(월별 총괄) 컬럼 매핑")
    if overall_raw.empty:
        st.warning("A(월별 총괄) 데이터가 없습니다. Parquet 업로드 또는 저장소에 Parquet 파일을 넣어주세요.")
        st.stop()

    colsA = [str(c) for c in overall_raw.columns]
    c_date = st.sidebar.selectbox("날짜 열", colsA, index=colsA.index(pick_best(colsA, ["날짜","Date","월"], colsA[0])))
    # 주택/산업 기본 열 시도
    c_cook = st.sidebar.selectbox("취사용 열", colsA, index=colsA.index(pick_best(colsA, ["취사용","주택","House","Cooking"], colsA[1])))
    # 존재하는 보조 열 자동발견
    extra_present = [c for c in CAND_EXTRA if c in colsA]
    st.sidebar.caption("대시보드 스택에 포함할 추가 열(있으면 자동 포함)")

    overall = overall_raw.copy()
    overall["날짜"] = pd.to_datetime(overall[c_date], errors="coerce")
    overall["취사용"] = overall[c_cook].apply(to_num)

    # 사용 가능한 사용열
    usage_cols = ["취사용"]
    for nm in extra_present:
        overall[nm] = overall[nm].apply(to_num)
        usage_cols.append(nm)
    usage_cols = [c for c in usage_cols if c in overall.columns]

    # ---------- B 컬럼 매핑 ----------
    st.sidebar.header("③ B(산업용 상세) 컬럼 매핑")
    if indetail_raw.empty:
        st.info("B(산업용 상세) 데이터가 없으면 ‘산업용 집중분석’ 탭은 표시만 됩니다.")
    colsB = [str(c) for c in indetail_raw.columns] if not indetail_raw.empty else []

    if colsB:
        b_date = st.sidebar.selectbox("B: 날짜(월)", colsB, index=colsB.index(pick_best(colsB, ["청구년월","사용월","년월","날짜","월"], colsB[0])))
        b_use  = st.sidebar.selectbox("B: 용도", colsB, index=colsB.index(pick_best(colsB, ["용도","Use","분류"], colsB[0])))
        b_ind  = st.sidebar.selectbox("B: 업종", colsB, index=colsB.index(pick_best(colsB, ["업종","Industry","업태","분류2"], colsB[0])))
        b_cus  = st.sidebar.selectbox("B: 고객명", colsB, index=colsB.index(pick_best(colsB, ["고객","고객명","거래처","사업장","업체"], colsB[0])))
        b_amt  = st.sidebar.selectbox("B: 사용량 열", colsB, index=colsB.index(pick_best(colsB, ["사용량","수량","NM3","Nm3","m3","MJ"], colsB[0])))

        def parse_month(x):
            s = str(x)
            for fmt in ["%Y-%m","%Y/%m","%Y%m","%Y.%m","%Y-%m-%d","%Y/%m/%d"]:
                try:
                    return pd.to_datetime(s, format=fmt).replace(day=1)
                except Exception:
                    pass
            return pd.to_datetime(s, errors="coerce")

        indetail = indetail_raw.copy()
        indetail["날짜"]   = pd.to_datetime(indetail[b_date].apply(parse_month), errors="coerce")
        indetail["용도"]   = indetail[b_use].astype(str).str.strip()
        indetail["업종"]   = indetail[b_ind].astype(str).str.strip()
        indetail["고객명"] = indetail[b_cus].astype(str).str.strip()
        indetail["사용량"] = pd.to_numeric(indetail[b_amt].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce").fillna(0)
    else:
        indetail = pd.DataFrame(columns=["날짜","용도","업종","고객명","사용량"])

    # ---------- 기간 범위 ----------
    st.title("📊 도시가스 판매량 분석 — 월/분기/반기/연간 + 산업용 업종/고객")
    date_min = pd.to_datetime(overall["날짜"]).min()
    date_max = pd.to_datetime(overall["날짜"]).max()
    default_start = pd.to_datetime(date_min or "2015-01-01")
    default_end   = pd.to_datetime(date_max or pd.Timestamp.today())

    _di = st.sidebar.date_input("기간", [default_start, default_end])
    if isinstance(_di, (list, tuple)) and len(_di) == 2:
        d1, d2 = _di
    else:
        d1, d2 = default_start, default_end

    # ---------- 탭 ----------
    tab0, tab1, tab2 = st.tabs(["🏠 대시보드","📚 집계","🏭 산업용 집중분석"])

    # ===== 탭0: 랜딩(연도×용도 스택) =====
    with tab0:
        st.subheader("연도별 용도 누적 스택")
        landing = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
        landing["연도"] = landing["날짜"].dt.year

        if not usage_cols:
            st.warning("표시할 사용열이 없습니다. A(월별 총괄)에서 ‘취사용’ 등 사용열을 확인하세요.")
        else:
            annual = landing.groupby("연도", as_index=False)[usage_cols].sum().sort_values("연도")
            fig0 = go.Figure()
            for col in usage_cols:
                fig0.add_trace(go.Bar(x=annual["연도"], y=annual[col], name=col))
            fig0.update_layout(
                barmode="stack", template="simple_white", height=420,
                xaxis=dict(title="Year"), yaxis=dict(title="사용량"),
                font=dict(family=FONT, size=13), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
            )
            st.plotly_chart(fig0, use_container_width=True, config={"displaylogo": False})
            st.dataframe(annual.set_index("연도").style.format("{:,.0f}"), use_container_width=True)

    # ===== 탭1: 집계 =====
    with tab1:
        st.subheader("집계 — 월/분기/반기/연간")
        gran = st.radio("집계 단위", ["월","분기","반기","연간"], horizontal=True, key="granularity")
        A = overall[(overall["날짜"]>=pd.to_datetime(d1)) & (overall["날짜"]<=pd.to_datetime(d2))].copy()
        A["Period"] = as_period_key(A["날짜"], gran)
        if not usage_cols:
            st.info("표시할 사용열이 없습니다.")
        else:
            sum_tbl = A.groupby("Period", as_index=False)[usage_cols].sum().sort_values("Period")
            left, right = st.columns([2,3])
            with left:
                st.dataframe(sum_tbl.style.format({c:"{:,.0f}" for c in usage_cols}), use_container_width=True)
            with right:
                fig = go.Figure()
                for col in usage_cols:
                    fig.add_trace(go.Bar(x=sum_tbl["Period"], y=sum_tbl[col], name=col))
                fig.update_layout(
                    barmode="group", template="simple_white", height=360,
                    xaxis=dict(title="Period"), yaxis=dict(title="사용량"),
                    font=dict(family=FONT, size=13)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # ===== 탭2: 산업용 집중분석 =====
    with tab2:
        st.subheader("산업용 집중분석 — 업종 히트맵 → 고객 Top-N/YoY")
        if indetail.empty:
            st.info("산업용 상세 데이터(B)가 없습니다. Parquet 업로드 또는 저장소에 `상품별판매량.parquet` 등을 넣어주세요.")
        else:
            # 산업용만 필터(파일에 다른 용도 섞여있을 수 있음)
            B = indetail.copy()
            if "용도" in B.columns:
                # '산업' 포함 문자열만 유지
                mask = B["용도"].astype(str).str.contains("산업", na=False)
                if mask.any():
                    B = B[mask]
            # 기간
            B = B[(B["날짜"]>=pd.to_datetime(d1)) & (B["날짜"]<=pd.to_datetime(d2))].copy()
            if B.empty:
                st.info("선택 기간에 데이터가 없습니다.")
            else:
                gran_focus = st.radio("기간 단위", ["월","분기","반기","연간"], horizontal=True, key="gran_focus")
                B["Period"] = as_period_key(B["날짜"], gran_focus)

                # 업종×기간 피벗 → 히트맵
                pivot = B.pivot_table(index="업종", columns="Period", values="사용량", aggfunc="sum").fillna(0)
                pivot = pivot[pivot.columns.sort_values()].sort_index()

                if pivot.empty:
                    st.info("업종×기간 피벗 결과가 비어 있습니다. 컬럼 매핑을 다시 확인하세요.")
                else:
                    Z = pivot.values
                    X = pivot.columns.tolist()
                    Y = pivot.index.tolist()
                    heat = go.Figure(data=go.Heatmap(
                        z=Z, x=X, y=Y, colorscale="Blues",
                        colorbar=dict(title="사용량"),
                        hovertemplate="업종=%{y}<br>기간=%{x}<br>사용량=%{z:,.0f}<extra></extra>"
                    ))
                    heat.update_layout(template="simple_white", height=560,
                                       xaxis=dict(title="Period"), yaxis=dict(title="업종"),
                                       font=dict(family=FONT, size=13), margin=dict(l=70, r=20, t=40, b=40))
                    clicked = plotly_events(heat, click_event=True, hover_event=False, select_event=False,
                                            override_height=560, override_width="100%")

                    # 셀 클릭/대체 선택 UI
                    if clicked:
                        sel_period = str(clicked[0].get("x"))
                        sel_ind = str(clicked[0].get("y"))
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            sel_ind = st.selectbox("업종 선택", Y)
                        with c2:
                            sel_period = st.selectbox("기간 선택", X)

                    # 선택 결과 테이블/막대
                    yo = yoy_compare(B[B["업종"]==sel_ind], ["업종","고객명"], "사용량", "Period", gran_focus)
                    yo_sel = yo[yo["Period"]==sel_period].copy().sort_values("사용량", ascending=False)

                    if yo_sel.empty:
                        st.info("선택된 업종/기간에 고객 데이터가 없습니다.")
                    else:
                        yo_sel["사용량"]   = yo_sel["사용량"].round(0)
                        yo_sel["전년동기"] = yo_sel["전년동기"].round(0)
                        yo_sel["증감"]     = yo_sel["증감"].round(0)
                        yo_sel["YoY(%)"]  = yo_sel["YoY(%)"].round(1)

                        top_n = st.slider("상위 N", 5, 100, 20, step=5)
                        view = yo_sel.head(top_n)[["고객명","사용량","전년동기","증감","YoY(%)"]].reset_index(drop=True)

                        g1, g2 = st.columns([1.4, 1.6])
                        with g1:
                            st.markdown(f"**선택 업종:** `{sel_ind}` · **선택 기간:** `{sel_period}`")
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

    # ---------- 사용된 원천 파일 ----------
    with st.expander("🔎 분석에 사용된 원천 파일"):
        if used_overall:
            st.write("A(월별 총괄): " + ", ".join(used_overall[:10]) + (" …" if len(used_overall) > 10 else ""))
        else:
            st.write("A(월별 총괄): (업로드/자동탐색 결과 없음)")
        if used_inds:
            st.write("B(산업용 상세): " + ", ".join(used_inds[:10]) + (" …" if len(used_inds) > 10 else ""))
        else:
            st.write("B(산업용 상세): (업로드/자동탐색 결과 없음)")
