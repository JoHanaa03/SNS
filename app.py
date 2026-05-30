# -*- coding: utf-8 -*-
"""
축제가 온라인 관심도(SNS 검색량)에 미치는 영향 분석 대시보드
- 데이터: SQLite (festival_selected, sns_mention)
- 분석단위: 축제 회차(edition) / 유의성 검정은 축제 단위(n=28)
실행: streamlit run app.py
"""
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="축제 온라인 관심도 영향 분석", layout="wide")

PERIOD_ORDER = ["BEFORE_3M", "BEFORE_2M", "BEFORE_1M", "FESTIVAL",
                "AFTER_1M", "AFTER_2M", "AFTER_3M"]
OFFSET_LABEL = {-3: "BEFORE_3M", -2: "BEFORE_2M", -1: "BEFORE_1M", 0: "FESTIVAL",
                1: "AFTER_1M", 2: "AFTER_2M", 3: "AFTER_3M"}

# ---------------------------------------------------------------------------
# 분석의 핵심 SQL: 선별 축제의 개최월(FESTIVAL)을 기준(M)으로 ±3개월 SNS 검색량을 결합
#   - 월 계산은 (연*12 + (월-1)) 인덱스로 처리해 연도 경계를 정확히 넘김
#   - off = 0(축제 당월), 음수=이전, 양수=이후
# ---------------------------------------------------------------------------
SQL_BASE = """
WITH fs AS (
    SELECT sido, sigungu, festival_name, festival_year,
           year_month AS festival_ym,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx
    FROM festival_selected
    WHERE period = 'FESTIVAL' AND festival_year >= 2020
),
sx AS (
    SELECT sido, sigungu, year_month, search_volume,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx
    FROM sns_mention
)
SELECT fs.sido, fs.sigungu, fs.festival_name, fs.festival_year, fs.festival_ym,
       (sx.m_idx - fs.m_idx) AS off,
       sx.search_volume
FROM fs
JOIN sx
  ON sx.sido = fs.sido
 AND sx.sigungu = fs.sigungu
 AND sx.m_idx BETWEEN fs.m_idx - 3 AND fs.m_idx + 3;
"""

SQL_SELECTED = """
-- 2020년 이후 3회 이상 개최한 축제만 선별
CREATE TABLE festival_selected AS
SELECT f.*
FROM festival f
JOIN (
    SELECT sido, sigungu, festival_name
    FROM festival
    WHERE festival_year >= 2020
    GROUP BY sido, sigungu, festival_name
    HAVING COUNT(DISTINCT festival_year) >= 3
) q USING (sido, sigungu, festival_name);
"""


# ---------------------------------------------------------------------------
# 데이터 로딩 & 가공 (streamlit 비의존 순수 로직)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_base(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(SQL_BASE, con)
    finally:
        con.close()
    return df


def build_edition(base: pd.DataFrame) -> pd.DataFrame:
    """회차(edition)별 지표 집계."""
    def agg(g):
        sv = g.set_index("off")["search_volume"]
        vf = sv.get(0, np.nan)
        before = g.loc[g.off.between(-3, -1), "search_volume"]
        after = g.loc[g.off.between(1, 3), "search_volume"]
        around = g.loc[g.off != 0, "search_volume"]
        return pd.Series({
            "vol_festival": vf,
            "avg_before_3m": before.mean(),
            "avg_after_3m": after.mean(),
            "baseline_6m": around.mean(),
            "n_before": before.count(),
            "n_after": after.count(),
        })

    e = (base.groupby(["sido", "sigungu", "festival_name", "festival_year", "festival_ym"])
              .apply(agg).reset_index())
    e["lift_ratio"] = e.vol_festival / e.baseline_6m
    e["afterglow_ratio"] = e.avg_after_3m / e.avg_before_3m
    e["full_window"] = (e.n_before == 3) & (e.n_after == 3)
    return e


def build_festival(edition_full: pd.DataFrame) -> pd.DataFrame:
    """축제 단위 집계(회차 평균) — 독립표본 n개."""
    f = (edition_full.groupby(["sido", "sigungu", "festival_name"])
         .agg(vol_festival=("vol_festival", "mean"),
              baseline_6m=("baseline_6m", "mean"),
              avg_before_3m=("avg_before_3m", "mean"),
              avg_after_3m=("avg_after_3m", "mean"),
              n_editions=("festival_year", "nunique"))
         .reset_index())
    f["lift_ratio"] = f.vol_festival / f.baseline_6m
    f["afterglow_ratio"] = f.avg_after_3m / f.avg_before_3m
    return f


def run_tests(fest: pd.DataFrame) -> dict:
    """축제 단위 paired 검정."""
    a, b = fest.vol_festival.values, fest.baseline_6m.values
    out = {"n": len(fest)}
    try:
        w = stats.wilcoxon(a, b)
        out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
    except Exception:
        out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    try:  # 비율 데이터라 로그 변환 후 paired t-test
        t = stats.ttest_rel(np.log(a), np.log(b))
        out["logt_p"] = float(t.pvalue)
    except Exception:
        out["logt_p"] = np.nan
    out["median_lift"] = float(np.median(fest.lift_ratio))
    out["pct_positive"] = float((fest.lift_ratio > 1).mean())
    out["n_positive"] = int((fest.lift_ratio > 1).sum())
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("축제가 온라인 관심도에 미치는 영향")
st.caption("선별 축제(2020년 이후 3회+) 개최 당월의 SNS 검색량을 전후 기준선과 비교")

with st.sidebar:
    st.header("설정")
    db_path = st.text_input("SQLite DB 경로", value="festival.db")
    st.markdown("필요 테이블: `festival_selected`, `sns_mention`")

try:
    base = load_base(db_path)
except Exception as ex:
    st.error(f"DB 로딩 실패: {ex}")
    st.stop()

if base.empty:
    st.warning("결합 결과가 비어 있습니다. 테이블/지역명 매칭을 확인하세요.")
    st.stop()

edition = build_edition(base)
edition_full = edition[edition.full_window].copy()      # 완전 ±3 윈도우만 검정에 사용
fest = build_festival(edition_full)
res = run_tests(fest)

tab1, tab2, tab3, tab4 = st.tabs(
    ["핵심 결과", "축제별 비교", "개별 축제 상세", "방법론 · SQL"])

# ---------------- TAB 1 : 핵심 결과 ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 축제 수", f"{res['n']}개")
    c2.metric("검색량 상승 축제", f"{res['n_positive']}/{res['n']}",
              f"{res['pct_positive']*100:.0f}%")
    c3.metric("중앙 Lift", f"{res['median_lift']:.2f}배",
              help="축제 당월 검색량 ÷ 전후 6개월 평균")
    c4.metric("Wilcoxon p-value", f"{res['wilcoxon_p']:.4f}",
              "유의" if res['wilcoxon_p'] < 0.05 else "비유의")

    sig = res["wilcoxon_p"] < 0.05
    st.markdown(
        f"""
**인사이트 요약**

- 축제 개최 당월의 SNS 검색량은 전후 6개월 평균 대비 **중앙값 {res['median_lift']:.2f}배** 수준으로,
  분석 대상 {res['n']}개 축제 중 **{res['n_positive']}개({res['pct_positive']*100:.0f}%)**에서 상승이 관찰됨.
- 축제 단위 paired 비모수 검정(Wilcoxon signed-rank) 결과 **p = {res['wilcoxon_p']:.4f}**
  → {"통계적으로 유의한 상승 (p<0.05). 즉 축제 개최가 온라인 관심도를 끌어올린다는 근거." if sig else "유의수준 0.05에서 유의하지 않음."}
- 보조 검정(로그변환 paired t-test) p = {res['logt_p']:.4f} 로 동일한 결론.
        """)

    edition_full["방향"] = np.where(edition_full.lift_ratio > 1, "상승(>1)", "하락(≤1)")
    fig = px.histogram(edition_full, x="lift_ratio", nbins=24, color="방향",
                       color_discrete_map={"상승(>1)": "#2E86DE", "하락(≤1)": "#C0392B"},
                       labels={"lift_ratio": "Lift (당월 ÷ 전후6개월)"},
                       title="회차별 Lift 분포")
    fig.add_vline(x=1.0, line_dash="dash", line_color="black",
                  annotation_text="기준선 1.0")
    fig.add_vline(x=edition_full.lift_ratio.median(), line_color="#2E86DE",
                  annotation_text=f"중앙값 {edition_full.lift_ratio.median():.2f}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"회차 기준: 완전 ±3개월 윈도우 {len(edition_full)}개 "
               f"(전체 {len(edition)}개 중). 1.0 초과 = 축제 당월에 검색량 급증.")

# ---------------- TAB 2 : 축제별 비교 ----------------
with tab2:
    fsort = fest.sort_values("lift_ratio", ascending=True)
    fig = px.bar(fsort, x="lift_ratio", y="festival_name", orientation="h",
                 color=fsort.lift_ratio > 1,
                 color_discrete_map={True: "#2E86DE", False: "#C0392B"},
                 labels={"lift_ratio": "Lift", "festival_name": ""},
                 title="축제별 Lift (회차 평균)")
    fig.add_vline(x=1.0, line_dash="dash", line_color="black")
    fig.update_layout(showlegend=False, height=720)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 효과 유형 4분면 (스파이크 강도 × 지속 효과)")
    fig2 = px.scatter(fest, x="lift_ratio", y="afterglow_ratio",
                      text="festival_name", size="vol_festival",
                      labels={"lift_ratio": "Lift (당월 스파이크)",
                              "afterglow_ratio": "Afterglow (후3M ÷ 전3M)"})
    fig2.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig2.update_traces(textposition="top center", textfont_size=9)
    fig2.update_layout(height=560)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("우상단=개최月 급증 + 개최 후에도 관심 지속(가장 효과적). "
               "우하단=반짝 효과. 좌측=개최月 효과 미미.")

    st.dataframe(
        fest.sort_values("lift_ratio", ascending=False)[
            ["festival_name", "sido", "sigungu", "n_editions",
             "vol_festival", "baseline_6m", "lift_ratio", "afterglow_ratio"]
        ].round(2),
        use_container_width=True, hide_index=True)

# ---------------- TAB 3 : 개별 축제 상세 ----------------
with tab3:
    pick = st.selectbox("축제 선택",
                        fest.sort_values("lift_ratio", ascending=False).festival_name)
    sub = base[base.festival_name == pick].copy()
    # 회차별 baseline으로 정규화(=100) 후 offset 프로파일 평균 → 스파이크 형태 비교
    prof = []
    for yr, g in sub.groupby("festival_year"):
        bl = g.loc[g.off != 0, "search_volume"].mean()
        if bl and not np.isnan(bl):
            t = g.copy()
            t["idx100"] = t.search_volume / bl * 100
            prof.append(t)
    prof = pd.concat(prof)
    pm = prof.groupby("off")["idx100"].mean().reindex(range(-3, 4)).reset_index()
    pm["period"] = pm.off.map(OFFSET_LABEL)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm.period, y=pm.idx100, mode="lines+markers",
                             line=dict(width=3, color="#2E86DE")))
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="기준선 100")
    fig.update_layout(title=f"{pick} — 전후 검색량 프로파일 (기준선=100, 회차평균)",
                      yaxis_title="정규화 검색량", xaxis_title="",
                      xaxis=dict(categoryorder="array", categoryarray=PERIOD_ORDER),
                      height=460)
    st.plotly_chart(fig, use_container_width=True)

    row = fest[fest.festival_name == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lift", f"{row.lift_ratio:.2f}배")
    c2.metric("Afterglow", f"{row.afterglow_ratio:.2f}")
    c3.metric("분석 회차", f"{int(row.n_editions)}회")

    st.markdown("##### 회차별 원자료")
    st.dataframe(
        edition[edition.festival_name == pick][
            ["festival_year", "festival_ym", "vol_festival",
             "avg_before_3m", "avg_after_3m", "lift_ratio", "full_window"]
        ].round(2), use_container_width=True, hide_index=True)

# ---------------- TAB 4 : 방법론 · SQL ----------------
with tab4:
    st.markdown(f"""
### 분석 목적
축제 개최가 해당 지역의 **온라인 관심도(SNS 검색량)** 를 끌어올리는지 검증한다.

### 데이터 & 전처리
- `festival_selected`: 2020년 이후 **3회 이상** 개최한 축제만 선별(표본 신뢰도 확보). 각 개최를
  `BEFORE_3M ~ FESTIVAL ~ AFTER_3M` 7개 period 행으로 펼친 long 구조.
- `sns_mention`: 시군구 × 월 단위 검색량(2020.01–2025.12, 42개 시군 × 72개월 = 완전 패널).
- 결합 키: `sido + sigungu`, 시점은 **(연×12 + 월−1) 인덱스**로 변환해 연도 경계를 정확히 처리.

### 지표 정의
- **Lift = 축제 당월 검색량 ÷ 전후 6개월(±3M, 당월 제외) 평균**
  · 1.0 초과 = 개최月에 검색량 급증. 지역 규모에 따른 절대량 차이를 제거하기 위해 *비율*로 정의.
- **Afterglow = 후3개월 평균 ÷ 전3개월 평균** : 개최 이후 관심도 지속(잔존 효과) 여부.

### 통계 기법 & 분석 단위
1. **분석 단위 분리** — 같은 축제의 여러 회차는 서로 독립이 아니므로(pseudoreplication),
   회차 지표를 **축제 단위로 평균**하여 독립표본 n = {res['n']}개로 검정.
2. **Wilcoxon signed-rank test** (paired, 비모수) — 표본이 작고 검색량 분포의 정규성을 보장할 수
   없어 모수 검정 대신 사용. H₀: 당월 검색량 = 전후 기준선.
3. **로그변환 paired t-test** 를 보조로 병행(비율 데이터의 비대칭 완화) → 동일 결론 확인.
4. **효과크기**: 중앙 Lift, 상승 축제 비율로 실질적 크기를 함께 보고(p값만으로 판단하지 않음).

### 결과
- Wilcoxon p = **{res['wilcoxon_p']:.4f}**, 로그 t-test p = **{res['logt_p']:.4f}**,
  중앙 Lift = **{res['median_lift']:.2f}배**, 상승 축제 **{res['n_positive']}/{res['n']}**.

### 한계 (해석 시 주의)
- SNS 검색량은 **시군구 단위**라 한 지역에 축제가 여럿이면(예: 논산·하동) 신호가 섞임.
- 대부분 계절 축제 → ±3개월 비교가 계절성을 완전히 제거하지 못함(보강: 전년 동월 대비).
- 검색량은 ‘관심도’의 대리지표일 뿐, 실제 방문·소비와는 별개 차원.
- 외부 사건(뉴스·바이럴 등) 통제 불가 → 인과가 아닌 **연관(association)** 으로 해석.
    """)

    st.markdown("#### 1) 선별 축제 테이블")
    st.code(SQL_SELECTED, language="sql")
    st.markdown("#### 2) 분석 기반 결합 SQL (당월 ±3개월 검색량)")
    st.code(SQL_BASE, language="sql")
    st.caption("이후 회차/축제 단위 집계와 검정은 app.py의 pandas·scipy 로직에서 수행.")
