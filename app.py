# -*- coding: utf-8 -*-
"""
축제가 온라인 관심도(SNS 검색량)에 미치는 영향 — 계절성 보정(이중차분) 대시보드

분석 엔진은 SQL이 담당한다:
  · 처치(개최 회차)와 반사실(코로나로 미개최였던 2020/2021 같은 달) anchor 생성
  · SNS 패널과 ±3개월 윈도우 결합
  · anchor별 완전윈도우 지표(Lift, Afterglow) 계산
  · 축제 단위 집계 / 전후 프로파일 집계
Python(scipy)은 통계 검정(Wilcoxon)만 수행한다.

유지 조건: ±3개월 윈도우, 2020년 이후 개최 축제 전체.
실행: streamlit run app.py
"""
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="축제 온라인 관심도 영향 (계절보정)", layout="wide")

PERIOD_ORDER = ["BEFORE_3M", "BEFORE_2M", "BEFORE_1M", "FESTIVAL",
                "AFTER_1M", "AFTER_2M", "AFTER_3M"]
OFFSET_LABEL = {-3: "BEFORE_3M", -2: "BEFORE_2M", -1: "BEFORE_1M", 0: "FESTIVAL",
                1: "AFTER_1M", 2: "AFTER_2M", 3: "AFTER_3M"}

# ===========================================================================
# 공통 CTE — 처치/반사실 anchor 생성 → SNS ±3개월 결합 → anchor별 윈도우 지표
#   · 월 인덱스 = 연*12 + (월-1) 로 연도 경계를 정확히 처리
#   · 반사실 = 같은 시군구·같은 달(month-of-year)의 2020·2021년(코로나 미개최)
# ===========================================================================
CTE = """
WITH fest_occ AS (                       -- (1) 처치 anchor: 실제 개최 회차
    SELECT sido, sigungu, festival_name,
           festival_year      AS anchor_year,
           year_month         AS anchor_ym,
           (year_month%100)   AS moy,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx,
           '처치(개최)'        AS kind
    FROM festival
    WHERE period = 'FESTIVAL' AND festival_year >= 2020
),
moy_list AS (                            -- (2) 축제별 개최 '월' 목록(반사실 기준)
    SELECT DISTINCT sido, sigungu, festival_name, moy FROM fest_occ
),
placebo_occ AS (                         -- (3) 반사실 anchor: 같은 달 × 2020/2021
    SELECT m.sido, m.sigungu, m.festival_name,
           py.y               AS anchor_year,
           py.y*100 + m.moy   AS anchor_ym,
           m.moy,
           py.y*12 + (m.moy - 1) AS m_idx,
           '반사실(미개최)'    AS kind
    FROM moy_list m
    CROSS JOIN (SELECT 2020 AS y UNION ALL SELECT 2021) py
),
anchors AS (                             -- (4) 처치 + 반사실 통합
    SELECT * FROM fest_occ
    UNION ALL
    SELECT * FROM placebo_occ
),
sx AS (                                  -- (5) SNS 패널에 월 인덱스 부여
    SELECT sido, sigungu, search_volume,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx
    FROM sns_mention
),
joined AS (                              -- (6) anchor ±3개월 윈도우로 SNS 결합
    SELECT a.sido, a.sigungu, a.festival_name, a.kind,
           a.anchor_year, a.anchor_ym,
           (s.m_idx - a.m_idx) AS off,
           s.search_volume     AS sv
    FROM anchors a
    JOIN sx s
      ON s.sido = a.sido AND s.sigungu = a.sigungu
     AND s.m_idx BETWEEN a.m_idx - 3 AND a.m_idx + 3
),
metrics AS (                             -- (7) anchor별 지표 + 윈도우 완전성 카운트
    SELECT sido, sigungu, festival_name, kind, anchor_year, anchor_ym,
           MAX(CASE WHEN off = 0 THEN sv END)                        AS vol_festival,
           AVG(CASE WHEN off BETWEEN -3 AND -1 THEN sv END)          AS avg_before,
           AVG(CASE WHEN off BETWEEN  1 AND  3 THEN sv END)          AS avg_after,
           AVG(CASE WHEN off <> 0 THEN sv END)                       AS baseline,
           COUNT(CASE WHEN off BETWEEN -3 AND -1 THEN 1 END)         AS nb,
           COUNT(CASE WHEN off BETWEEN  1 AND  3 THEN 1 END)         AS na,
           MAX(CASE WHEN off = 0 THEN 1 ELSE 0 END)                  AS has_m
    FROM joined
    GROUP BY sido, sigungu, festival_name, kind, anchor_year, anchor_ym
),
mfull AS (                               -- (8) 완전 ±3개월 윈도우만 + 지표 계산
    SELECT *,
           1.0 * vol_festival / baseline   AS lift,
           1.0 * avg_after    / avg_before AS afterglow
    FROM metrics
    WHERE nb = 3 AND na = 3 AND has_m = 1
)
"""

# (A) 축제 단위 집계: 처치 평균 vs 반사실 평균  →  Net 효과 산출 기반
SQL_FESTIVAL = CTE + """
SELECT sido, sigungu, festival_name,
   COUNT(CASE WHEN kind = '처치(개최)'   THEN 1 END) AS n_editions,
   COUNT(CASE WHEN kind LIKE '반사실%'    THEN 1 END) AS n_placebo,
   AVG(CASE WHEN kind = '처치(개최)' THEN lift       END) AS lift,
   AVG(CASE WHEN kind LIKE '반사실%'  THEN lift       END) AS plac_lift,
   AVG(CASE WHEN kind = '처치(개최)' THEN afterglow  END) AS afterglow,
   AVG(CASE WHEN kind LIKE '반사실%'  THEN afterglow  END) AS plac_afterglow,
   AVG(CASE WHEN kind = '처치(개최)' THEN vol_festival END) AS vol_festival,
   AVG(CASE WHEN kind = '처치(개최)' THEN baseline   END) AS baseline
FROM mfull
GROUP BY sido, sigungu, festival_name;
"""

# (B) anchor(회차/반사실) 단위 원자료 — 개별 축제 상세 탭용
SQL_EDITION = CTE + """
SELECT sido, sigungu, festival_name, kind, anchor_year, anchor_ym,
       vol_festival, avg_before, avg_after, baseline, lift, afterglow
FROM mfull
ORDER BY festival_name, kind DESC, anchor_year;
"""

# (C) 전후 프로파일: 처치 회차를 각자의 baseline(=100)으로 정규화해 offset별 평균
SQL_PROFILE = CTE + """
, treat_full AS (
    SELECT sido, sigungu, festival_name, anchor_ym, baseline
    FROM mfull WHERE kind = '처치(개최)'
)
SELECT j.festival_name, j.off,
       AVG(100.0 * j.sv / t.baseline) AS idx100
FROM joined j
JOIN treat_full t
  ON t.sido = j.sido AND t.sigungu = j.sigungu
 AND t.festival_name = j.festival_name AND t.anchor_ym = j.anchor_ym
WHERE j.kind = '처치(개최)'
GROUP BY j.festival_name, j.off;
"""

# (D) 보조: SNS 패널 커버리지(메서드 탭 표시용)
SQL_COVERAGE = """
SELECT COUNT(DISTINCT sido || sigungu)            AS n_region,
       COUNT(DISTINCT year_month)                 AS n_month,
       MIN(year_month) AS ym_min, MAX(year_month) AS ym_max,
       COUNT(*)                                   AS n_rows
FROM sns_mention;
"""


@st.cache_data(show_spinner=False)
def run_sql(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        F = pd.read_sql(SQL_FESTIVAL, con)
        E = pd.read_sql(SQL_EDITION, con)
        P = pd.read_sql(SQL_PROFILE, con)
        C = pd.read_sql(SQL_COVERAGE, con).iloc[0]
    finally:
        con.close()
    F["net_lift"] = F.lift - F.plac_lift
    F["net_afterglow"] = F.afterglow - F.plac_afterglow
    return F, E, P, C


def paired_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    try:
        w = stats.wilcoxon(a, b)
        return float(w.statistic), float(w.pvalue), len(a)
    except Exception:
        return np.nan, np.nan, len(a)


# ===========================================================================
# UI
# ===========================================================================
st.title("축제가 온라인 관심도에 미치는 실질 영향")
st.caption("계절성을 보정한 이중차분(DiD): 코로나로 축제가 없던 2020·2021년 같은 달을 대조군으로 사용 "
           "· ±3개월 윈도우 · 2020년 이후 개최 축제 전체 · 분석 엔진은 SQL")

with st.sidebar:
    st.header("설정")
    db_path = st.text_input("SQLite DB 경로", value="festival.db")
    st.markdown("필요 테이블: `festival`, `sns_mention`")
    st.divider()
    st.markdown("**지표 정의**")
    st.markdown(
        "- **Lift**: 축제月 ÷ 전후 6개월 평균\n"
        "- **반사실 Lift**: 같은 달(미개최 연)의 Lift\n"
        "- **Net Lift = Lift − 반사실 Lift**\n"
        "  → 계절수요를 뺀 *순수 축제 효과*\n"
        "- **Afterglow**: 후3M ÷ 전3M (지속효과)\n"
        "- **Net Afterglow**: 계절보정 지속효과")

try:
    F, E, P, C = run_sql(db_path)
except Exception as ex:
    st.error(f"DB 로딩/분석 실패: {ex}")
    st.stop()

if F.empty:
    st.warning("결합 결과가 비어 있습니다. 테이블/지역명 매칭을 확인하세요.")
    st.stop()

V = F.dropna(subset=["net_lift", "plac_lift"]).copy()

ls, lp, ln = paired_test(V.lift, V.plac_lift)
as_, ap, an = paired_test(V.afterglow, V.plac_afterglow)
raw_s, raw_p, raw_n = paired_test(F.vol_festival, F.baseline)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["핵심 결과", "Top5 vs Bottom5", "계절보정 효과", "효과 유형 분류", "개별 축제", "방법론 · SQL"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("계절성 보정 전 vs 후")
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 축제 수", f"{len(V)}개", help="반사실(2020/2021 같은달) 확보 축제")
    c2.metric("단기효과 Net Lift (중앙)", f"{V.net_lift.median():+.2f}",
              f"양(+) 축제 {int((V.net_lift>0).sum())}/{len(V)}")
    c3.metric("지속효과 Net Afterglow (중앙)", f"{V.net_afterglow.median():+.2f}",
              f"양(+) 축제 {int((V.net_afterglow>0).sum())}/{len(V)}")

    c4, c5, c6 = st.columns(3)
    c4.metric("보정 전 Lift (중앙)", f"{F.lift.median():.2f}배")
    c5.metric("단기효과 검정 p", f"{lp:.4f}", "유의" if lp < 0.05 else "비유의")
    c6.metric("지속효과 검정 p", f"{ap:.4f}", "유의" if ap < 0.05 else "비유의")

    sig_l, sig_a = lp < 0.05, ap < 0.05
    st.markdown(f"""
**핵심 인사이트**

1. **계절성을 빼면 효과는 작아진다.** 보정 전 Lift 중앙값은 {F.lift.median():.2f}배지만,
   계절수요(반사실)를 제거한 **Net Lift 중앙값은 {V.net_lift.median():+.2f}** 수준.
   기존 분석의 상당 부분은 *축제 효과가 아니라 그 지역·그 계절의 자연 검색 증가*였다.

2. **단기 관심도 상승은 {"유의하다" if sig_l else "유의하지 않다"}** (Wilcoxon 처치 vs 반사실, p={lp:.4f}).
   {int((V.net_lift>0).sum())}/{len(V)} 축제가 계절 보정 후에도 양(+)의 순효과.

3. **지속적 인지도 상승 근거는 {"있다" if sig_a else "없다"}** (Net Afterglow p={ap:.4f}).
   {"개최 후에도 검색이 베이스라인보다 높게 유지됨." if sig_a else "개최 후 관심도가 베이스라인 이상으로 유지된다는 증거는 없음 → '반짝 효과'에 가깝다."}

> 결론: 축제는 {"개최 당월의 온라인 관심도를 (작지만 유의하게) 끌어올린다" if sig_l else "개최 당월 관심도에 뚜렷한 순효과를 주지 못한다"}.
> 다만 {"지속적인 지역 인지도 상승으로 이어진다는 근거는 약하다." if not sig_a else "그 효과가 개최 이후에도 일정 부분 지속된다."}
""")

    comp = V.sort_values("net_lift", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp.festival_name, x=comp.lift - 1, name="보정 전 (Lift−1)",
                         orientation="h", marker_color="#BDC3C7"))
    fig.add_trace(go.Bar(y=comp.festival_name, x=comp.net_lift, name="Net Lift (계절보정)",
                         orientation="h", marker_color="#2E86DE"))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="overlay", height=820, title="보정 전 효과 vs 계절보정 순효과",
                      xaxis_title="효과 크기 (0 = 무효과)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("회색이 길고 파랑이 짧거나 음수면, 보였던 효과가 사실은 계절수요였다는 뜻.")

# ---------------- TAB 2 : Top5 vs Bottom5 ----------------
with tab2:
    st.subheader("Net Lift 기준 Top5 vs Bottom5 한눈에 비교")
    k = min(5, len(V) // 2)
    top = V.sort_values("net_lift", ascending=False).head(k).copy()
    bot = V.sort_values("net_lift").head(k).copy()
    top["grp"], bot["grp"] = "Top5 (효과 큼)", "Bottom5 (효과 없음/역효과)"
    tb = pd.concat([bot.sort_values("net_lift"), top.sort_values("net_lift")])
    tb["label"] = tb.festival_name

    # 처치 Lift vs 반사실 Lift를 나란히 → 왜 net이 그렇게 나오는지 한눈에
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tb.label, x=tb.lift, name="처치 Lift (개최 연)",
                         orientation="h", marker_color="#2E86DE",
                         text=tb.lift.round(2), textposition="outside"))
    fig.add_trace(go.Bar(y=tb.label, x=tb.plac_lift, name="반사실 Lift (미개최 연 같은 달)",
                         orientation="h", marker_color="#E59866",
                         text=tb.plac_lift.round(2), textposition="outside"))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                  annotation_text="기준선 1.0")
    fig.update_layout(barmode="group", height=620,
                      title="처치 vs 반사실 Lift — 막대 차이가 곧 순효과(Net Lift)",
                      xaxis_title="Lift", legend=dict(orientation="h"),
                      yaxis=dict(title=""))
    # Top/Bottom 경계선
    fig.add_hline(y=k - 0.5, line_dash="dot", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("파랑(개최)이 주황(미개최)보다 길수록 순수 축제효과 큼. "
               "Bottom5는 주황이 더 길어 = 축제 없이도 그 달에 원래 검색이 높은 '계절 착시'.")

    # Net Lift / Net Afterglow 요약 비교
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=tb.label, y=tb.net_lift, name="Net Lift",
                          marker_color=np.where(tb.net_lift > 0, "#2E86DE", "#C0392B")))
    fig2.add_trace(go.Scatter(x=tb.label, y=tb.net_afterglow, name="Net Afterglow",
                              mode="markers", marker=dict(size=11, color="#8E44AD",
                              symbol="diamond")))
    fig2.add_hline(y=0, line_color="black")
    fig2.update_layout(height=420, title="순효과 비교: 단기(Net Lift, 막대) + 지속(Net Afterglow, 점)",
                       yaxis_title="순효과", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Top5 — 순수 축제효과 상위**")
        st.dataframe(top.sort_values("net_lift", ascending=False)[
            ["festival_name", "lift", "plac_lift", "net_lift", "net_afterglow"]
        ].round(3), use_container_width=True, hide_index=True)
    with cc2:
        st.markdown("**Bottom5 — 효과 없음/계절 착시**")
        st.dataframe(bot.sort_values("net_lift")[
            ["festival_name", "lift", "plac_lift", "net_lift", "net_afterglow"]
        ].round(3), use_container_width=True, hide_index=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Net Lift — 계절수요를 제거한 순수 축제 효과")
    s = V.sort_values("net_lift")
    fig = px.bar(s, x="net_lift", y="festival_name", orientation="h",
                 color=s.net_lift > 0,
                 color_discrete_map={True: "#2E86DE", False: "#C0392B"},
                 labels={"net_lift": "Net Lift (축제 − 반사실)", "festival_name": ""})
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(showlegend=False, height=820)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("처치 vs 반사실 직접 비교")
    fig2 = px.scatter(V, x="plac_lift", y="lift", text="festival_name",
                      labels={"plac_lift": "반사실 Lift (미개최 연 같은 달)",
                              "lift": "처치 Lift (개최 연)"})
    lim = [min(V.plac_lift.min(), V.lift.min()) * 0.95,
           max(V.plac_lift.max(), V.lift.max()) * 1.05]
    fig2.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                              line=dict(dash="dash", color="gray"), name="y=x"))
    fig2.update_traces(textposition="top center", textfont_size=8,
                       selector=dict(mode="markers+text"))
    fig2.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("대각선 위쪽 = 축제 연도가 미개최 연도보다 검색이 더 튐(순효과 +).")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("효과 유형 4분면: 단기 순효과 × 지속 순효과")
    fig = px.scatter(V, x="net_lift", y="net_afterglow", text="festival_name",
                     size=V.vol_festival, color="net_lift",
                     color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     labels={"net_lift": "Net Lift (단기 순효과)",
                             "net_afterglow": "Net Afterglow (지속 순효과)"})
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.update_layout(height=640)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
- **우상단**: 개최月 관심 급증 + 개최 후에도 베이스라인 상승 → *지역 인지도에 실질 기여* (최우선 후보)
- **우하단**: 개최月엔 튀지만 곧 원위치 → *반짝 효과*
- **좌측(음수)**: 계절수요를 빼면 순효과 없음/역효과 → *효과 좋은 축제로 오선정 주의*
""")
    st.dataframe(
        V.sort_values("net_lift", ascending=False)[
            ["festival_name", "sido", "sigungu", "n_editions",
             "lift", "plac_lift", "net_lift", "afterglow", "plac_afterglow", "net_afterglow"]
        ].round(3), use_container_width=True, hide_index=True)

# ---------------- TAB 5 ----------------
with tab5:
    pick = st.selectbox("축제 선택", V.sort_values("net_lift", ascending=False).festival_name)
    row = F[F.festival_name == pick].iloc[0]

    prof = P[P.festival_name == pick].set_index("off")["idx100"].to_dict()
    figp = go.Figure()
    figp.add_trace(go.Scatter(
        x=[OFFSET_LABEL[o] for o in range(-3, 4)],
        y=[prof.get(o) for o in range(-3, 4)],
        mode="lines+markers", line=dict(width=3, color="#2E86DE"),
        name="개최 연(처치)"))
    figp.add_hline(y=100, line_dash="dash", line_color="gray",
                   annotation_text="기준선 100")
    figp.update_layout(
        title=f"{pick} — 개최 전후 검색량 프로파일 (기준선=100, 회차 평균)",
        yaxis_title="정규화 검색량", height=440,
        xaxis=dict(categoryorder="array", categoryarray=PERIOD_ORDER))
    st.plotly_chart(figp, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Lift", f"{row.net_lift:+.2f}",
              help=f"처치 {row.lift:.2f} − 반사실 {row.plac_lift:.2f}")
    c2.metric("Net Afterglow", f"{row.net_afterglow:+.2f}")
    c3.metric("개최 회차(완전윈도우)", f"{int(row.n_editions)}회")
    c4.metric("반사실 표본", f"{int(row.n_placebo)}개")

    st.markdown("##### 회차/반사실 원자료 (SQL 결과)")
    st.dataframe(
        E[E.festival_name == pick][
            ["kind", "anchor_year", "anchor_ym", "vol_festival",
             "avg_before", "avg_after", "baseline", "lift", "afterglow"]
        ].round(2), use_container_width=True, hide_index=True)

# ---------------- TAB 6 : 방법론 · SQL ----------------
with tab6:
    st.markdown(f"""
### 분석 목적
축제 개최가 해당 지역의 **온라인 관심도(SNS 검색량)** 를 *계절수요를 넘어서* 끌어올리는지,
또 그 효과가 **개최 이후에도 지속**되어 실질적 지역 인지도 상승으로 이어지는지 검증한다.

### 왜 기존 방식을 보정했나 (문제 정의)
대부분 매년 같은 시기에 열리는 **계절 축제**라, "축제月 vs ±3개월" 단순 비교는
*축제 효과* 와 *그 지역·그 계절의 자연 검색 수요* 를 분리하지 못한다.
보정 전 Lift 중앙값({F.lift.median():.2f}배)의 상당 부분이 계절성에서 비롯됨이 확인되었다.

### 식별 전략 — 코로나 자연실험 (이중차분, DiD)
분석 대상 축제 전부가 **2020·2021년에는 코로나로 미개최**였고, SNS 패널은 2020.01부터 존재한다.
같은 시군구·같은 달의 2020/2021 값은 **'축제가 없었던 같은 시즌'** 반사실(counterfactual)이 된다.
- **처치**: 개최 연도 개최月 ±3개월 윈도우 지표 · **반사실**: 같은 달·미개최 연도(2020/2021) 동일 지표
- **순효과** `Net = 처치 − 반사실` → 공통 계절성이 차감됨

### 지표 정의 (±3개월 윈도우 유지)
- **Lift** = 축제月 ÷ 전후 6개월(±3M, 당월 제외) 평균 → 단기 관심도 스파이크
- **Afterglow** = 후3개월 ÷ 전3개월 → 개최 후 관심 지속(인지도 잔존)
- **Net Lift / Net Afterglow** = 처치 − 반사실

### 분석 구현 (SQL 중심)
윈도우 결합·집계·반사실 비교는 모두 **SQL**에서 수행하고, Python은 **Wilcoxon 검정만** 담당한다.
SQL 파이프라인: `fest_occ`(처치) → `placebo_occ`(반사실) → `anchors`(통합)
→ `joined`(±3개월 SNS 결합) → `metrics`(anchor 지표·윈도우 완전성) → `mfull`(완전윈도우)
→ 축제 단위 집계.

### 통계 기법 & 분석 단위
1. **분석 단위 분리**: 한 축제의 여러 회차는 비독립(pseudoreplication) → 회차 지표를
   **축제 단위로 평균**해 독립표본 n = {len(V)}개로 검정.
2. **Wilcoxon signed-rank (paired, 비모수)**: 처치 vs 반사실. 소표본·비정규에 강건.
3. 지속효과도 Afterglow에 동일 검정. 4. p값과 **효과크기(Net 중앙값·양(+) 비율)** 병기.

### 결과 요약
- 단기효과: Net Lift 중앙 **{V.net_lift.median():+.2f}**, p = **{lp:.4f}** ({"유의" if lp<0.05 else "비유의"}),
  양(+) {int((V.net_lift>0).sum())}/{len(V)}.
- 지속효과: Net Afterglow 중앙 **{V.net_afterglow.median():+.2f}**, p = **{ap:.4f}** ({"유의" if ap<0.05 else "비유의"}).
- (참고) 계절 미보정 raw 검정 p = {raw_p:.4f} → 보정 시 효과가 보수적으로 줄어듦.
- SNS 패널: {int(C.n_region)}개 시군 × {int(C.n_month)}개월 ({int(C.ym_min)}–{int(C.ym_max)}), {int(C.n_rows)}행.

### 한계
- 반사실이 **코로나 시기(2020/2021)** → 팬데믹발 이동·검색 위축이 대조군에 섞여 효과를 다소 과대평가할 수 있음.
- SNS 검색량은 **시군구 단위** → 한 지역 다축제(논산·하동 등)는 신호 혼입.
- 검색량은 '관심도'의 대리지표 → 실제 방문·소비와는 별개.
- 인과적 식별의 *근사*. 외부 사건(뉴스·바이럴) 완전 통제 불가 → 강한 연관으로 해석.
""")
    st.markdown("#### (A) 축제 단위 집계 — 처치 vs 반사실 (메인 분석 SQL)")
    st.code(SQL_FESTIVAL, language="sql")
    st.markdown("#### (B) anchor(회차/반사실) 원자료 SQL")
    st.code(SQL_EDITION, language="sql")
    st.markdown("#### (C) 전후 프로파일 정규화 SQL")
    st.code(SQL_PROFILE, language="sql")
    st.markdown("#### (D) SNS 패널 커버리지 SQL")
    st.code(SQL_COVERAGE, language="sql")
    st.caption("Python(scipy)은 net = 처치 − 반사실 계산과 Wilcoxon/효과크기 산출만 담당.")
