# -*- coding: utf-8 -*-
"""
축제가 온라인 관심도(SNS 검색량)에 미치는 영향 분석 대시보드
스토리: 전체적으로 '반짝효과' → 단기+장기 모두 양(+)인 축제 선별(기준B)
실행: streamlit run app.py
"""
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="축제 온라인 관심도 분석", layout="wide")

PERIOD_ORDER = ["BEFORE_3M","BEFORE_2M","BEFORE_1M","FESTIVAL","AFTER_1M","AFTER_2M","AFTER_3M"]
OFFSET_LABEL = {-3:"BEFORE_3M",-2:"BEFORE_2M",-1:"BEFORE_1M",0:"FESTIVAL",
                1:"AFTER_1M",2:"AFTER_2M",3:"AFTER_3M"}

# ── 핵심 분석 SQL ────────────────────────────────────────────────────────────
# (1) 처치/반사실 이중차분 → 축제 단위 집계
# 핵심 아이디어:
#   · 처치(treatment): 2020년 이후 실제 개최 회차
#   · 반사실(placebo): 같은 시군구·같은 달, 코로나 미개최 연도(2020·2021)
#   · ±3개월 월 인덱스(연*12+월-1)로 연도 경계를 정확히 처리
#   · 완전 윈도우(전후 각 3개월 모두 존재)만 사용
SQL_FESTIVAL = """
WITH fest_occ AS (          -- 처치 anchor: 실제 개최 회차
    SELECT sido, sigungu, festival_name,
           festival_year                              AS anchor_year,
           year_month                                 AS anchor_ym,
           (year_month % 100)                         AS moy,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx,
           '처치'                                      AS kind
    FROM festival
    WHERE period = 'FESTIVAL' AND festival_year >= 2020
),
placebo_occ AS (            -- 반사실 anchor: 같은 달 × 미개최 연도(2020·2021)
    SELECT DISTINCT f.sido, f.sigungu, f.festival_name,
           py.y                     AS anchor_year,
           py.y*100 + f.moy         AS anchor_ym,
           f.moy,
           py.y*12 + (f.moy - 1)   AS m_idx,
           '반사실'                  AS kind
    FROM fest_occ f
    CROSS JOIN (SELECT 2020 AS y UNION ALL SELECT 2021) py
),
anchors AS (SELECT * FROM fest_occ UNION ALL SELECT * FROM placebo_occ),
joined AS (                 -- 각 anchor의 ±3개월 SNS 검색량 결합
    SELECT a.sido, a.sigungu, a.festival_name, a.kind,
           a.anchor_year, a.anchor_ym,
           (s.m_idx - a.m_idx)                         AS off,
           s.search_volume                              AS sv
    FROM anchors a
    JOIN (SELECT sido, sigungu, search_volume,
                 (year_month/100)*12 + (year_month%100 - 1) AS m_idx
          FROM sns_mention) s
      ON s.sido = a.sido AND s.sigungu = a.sigungu
     AND s.m_idx BETWEEN a.m_idx - 3 AND a.m_idx + 3
),
agg AS (                    -- anchor별 지표 집계 (완전 윈도우 필터 포함)
    SELECT sido, sigungu, festival_name, kind, anchor_year, anchor_ym,
           MAX(CASE WHEN off = 0               THEN sv END)     AS vol_festival,
           AVG(CASE WHEN off BETWEEN -3 AND -1 THEN sv END)     AS avg_before,
           AVG(CASE WHEN off BETWEEN  1 AND  3 THEN sv END)     AS avg_after,
           AVG(CASE WHEN off <> 0              THEN sv END)     AS baseline,
           COUNT(CASE WHEN off BETWEEN -3 AND -1 THEN 1 END)    AS nb,
           COUNT(CASE WHEN off BETWEEN  1 AND  3 THEN 1 END)    AS na
    FROM joined
    GROUP BY sido, sigungu, festival_name, kind, anchor_year, anchor_ym
    HAVING nb = 3 AND na = 3
       AND MAX(CASE WHEN off = 0 THEN 1 ELSE 0 END) = 1
),
anchor_metrics AS (         -- Lift·Afterglow 파생
    SELECT *,
           1.0 * vol_festival / baseline  AS lift,
           1.0 * avg_after   / avg_before AS afterglow
    FROM agg
)
-- 축제 단위: 처치 평균 vs 반사실 평균
SELECT sido, sigungu, festival_name,
       COUNT(CASE WHEN kind = '처치'  THEN 1 END)              AS n_editions,
       COUNT(CASE WHEN kind = '반사실' THEN 1 END)              AS n_placebo,
       AVG(CASE WHEN kind = '처치'  THEN lift       END)       AS lift,
       AVG(CASE WHEN kind = '반사실' THEN lift       END)       AS plac_lift,
       AVG(CASE WHEN kind = '처치'  THEN afterglow  END)       AS afterglow,
       AVG(CASE WHEN kind = '반사실' THEN afterglow  END)       AS plac_afterglow,
       AVG(CASE WHEN kind = '처치'  THEN vol_festival END)     AS vol_festival,
       AVG(CASE WHEN kind = '처치'  THEN baseline    END)      AS baseline
FROM anchor_metrics
GROUP BY sido, sigungu, festival_name;
"""

# (2) 전후 프로파일: 처치 회차를 각자의 baseline=100으로 정규화 후 offset별 평균
SQL_PROFILE = """
WITH fest_occ AS (
    SELECT sido, sigungu, festival_name, year_month AS anchor_ym,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx
    FROM festival WHERE period = 'FESTIVAL' AND festival_year >= 2020
),
joined AS (
    SELECT f.festival_name, f.anchor_ym,
           AVG(CASE WHEN s.m_idx <> f.m_idx THEN s.search_volume END) AS baseline,
           COUNT(CASE WHEN s.m_idx < f.m_idx THEN 1 END)              AS nb,
           COUNT(CASE WHEN s.m_idx > f.m_idx THEN 1 END)              AS na
    FROM fest_occ f
    JOIN (SELECT sido, sigungu, search_volume,
                 (year_month/100)*12 + (year_month%100 - 1) AS m_idx
          FROM sns_mention) s
      ON s.sido = f.sido AND s.sigungu = f.sigungu
     AND s.m_idx BETWEEN f.m_idx - 3 AND f.m_idx + 3
    GROUP BY f.festival_name, f.anchor_ym
    HAVING nb = 3 AND na = 3
),
offset_sv AS (
    SELECT f.festival_name, f.anchor_ym,
           (s.m_idx - f.m_idx) AS off,
           s.search_volume      AS sv
    FROM fest_occ f
    JOIN joined j ON j.festival_name = f.festival_name AND j.anchor_ym = f.anchor_ym
    JOIN (SELECT sido, sigungu, search_volume,
                 (year_month/100)*12 + (year_month%100 - 1) AS m_idx
          FROM sns_mention) s
      ON s.sido = f.sido AND s.sigungu = f.sigungu
     AND s.m_idx BETWEEN f.m_idx - 3 AND f.m_idx + 3
)
SELECT o.festival_name, o.off,
       AVG(100.0 * o.sv / j.baseline) AS idx100
FROM offset_sv o
JOIN joined j ON j.festival_name = o.festival_name AND j.anchor_ym = o.anchor_ym
GROUP BY o.festival_name, o.off;
"""


@st.cache_data(show_spinner=False)
def load(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        F = pd.read_sql(SQL_FESTIVAL, con)
        P = pd.read_sql(SQL_PROFILE, con)
    finally:
        con.close()
    F["net_lift"]      = F.lift      - F.plac_lift
    F["net_afterglow"] = F.afterglow - F.plac_afterglow
    return F, P


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    try:
        w = stats.wilcoxon(a[m], b[m])
        return float(w.pvalue), int(m.sum())
    except Exception:
        return np.nan, int(m.sum())


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("축제가 온라인 관심도에 미치는 영향")
st.caption("이중차분(DiD) · ±3개월 윈도우 · 2020년 이후 개최 축제 전체 · 코로나 미개최 연도를 대조군으로 계절성 보정")

with st.sidebar:
    st.header("설정")
    db_path = st.text_input("SQLite DB 경로", value="festival.db")
    st.divider()
    st.markdown("""
**지표**
- **Lift** = 축제月 ÷ 전후6개월 평균
- **Afterglow** = 후3M 평균 ÷ 전3M 평균
- **Net = 처치 − 반사실** (계절보정)
    
**선별 기준(B)**
- Net Lift > 0
- Net Afterglow > 0
""")

try:
    F, P = load(db_path)
except Exception as ex:
    st.error(f"분석 실패: {ex}")
    st.stop()

V = F.dropna(subset=["net_lift", "net_afterglow"]).copy()
SEL = V[(V.net_lift > 0) & (V.net_afterglow > 0)].sort_values("net_afterglow", ascending=False)

lp, ln = wilcoxon(V.lift, V.plac_lift)          # 단기효과 검정
ap, an = wilcoxon(V.afterglow, V.plac_afterglow) # 장기효과 검정

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["전체 결과", "Top5 vs Bottom5", "선별 축제", "개별 축제", "방법론 · SQL"])

# ── TAB 1: 전체 결과 ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("분석 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 축제", f"{len(V)}개")
    c2.metric("단기효과(Net Lift) p", f"{lp:.4f}", "유의 ✓" if lp < 0.05 else "비유의")
    c3.metric("장기효과(Net Afterglow) p", f"{ap:.4f}", "유의" if ap < 0.05 else "비유의 ✗")
    c4.metric("단기+장기 모두 양(+)", f"{len(SEL)}/{len(V)}개")

    sig_l, sig_a = lp < 0.05, ap < 0.05
    st.info(f"""
**핵심 결론**

- 축제는 개최 당월의 온라인 관심도를 **유의하게 끌어올린다** (p={lp:.4f}). \
Net Lift 중앙값 **+{V.net_lift.median():.2f}**, {int((V.net_lift>0).sum())}/{len(V)} 축제에서 양(+).
- 그러나 개최 이후 관심도가 지속적으로 상승한다는 근거는 **없다** (p={ap:.4f}). \
대부분의 축제는 **'반짝 효과'**에 그친다.
- 그 중 **단기·장기 모두 양(+)인 {len(SEL)}개 축제**는 다음 단계(방문·소비 종합분석)의 후보로 선별.
""")

    # 전체 Net Lift / Net Afterglow 산점도 (4분면)
    V["선별여부"] = V.apply(lambda r: "선별(기준B)" if r.net_lift > 0 and r.net_afterglow > 0
                            else ("단기만" if r.net_lift > 0 else
                                  ("장기만" if r.net_afterglow > 0 else "효과없음")), axis=1)
    color_map = {"선별(기준B)":"#2E86DE","단기만":"#85C1E9","장기만":"#A9DFBF","효과없음":"#E5E7E9"}

    fig = px.scatter(V, x="net_lift", y="net_afterglow", text="festival_name",
                     color="선별여부", color_discrete_map=color_map,
                     size="vol_festival",
                     labels={"net_lift":"Net Lift (단기 순효과)","net_afterglow":"Net Afterglow (장기 순효과)"})
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.update_layout(height=580, title="전체 축제 효과 분포 (우상단 = 선별)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("우상단: 단기+장기 모두 양(+) → 선별 대상 | 우하단: 반짝효과 | 좌측: 계절착시")

    # 보정 전/후 비교
    st.subheader("계절 보정 전·후 Lift 비교")
    comp = V.sort_values("net_lift", ascending=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=comp.festival_name, x=comp.lift - 1, name="보정 전 (Lift−1)",
                          orientation="h", marker_color="#D5D8DC"))
    fig2.add_trace(go.Bar(y=comp.festival_name, x=comp.net_lift, name="Net Lift (계절보정)",
                          orientation="h",
                          marker_color=["#2E86DE" if v > 0 else "#C0392B" for v in comp.net_lift]))
    fig2.add_vline(x=0, line_color="black")
    fig2.update_layout(barmode="overlay", height=820,
                       xaxis_title="효과 크기", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("회색이 길고 색이 짧거나 음수면 → 보였던 효과의 상당 부분이 계절수요였다는 뜻.")

# ── TAB 2: Top5 vs Bottom5 (net_afterglow 기준) ──────────────────────────────
with tab2:
    st.subheader("장기효과(Net Afterglow) 기준 Top5 vs Bottom5")
    st.caption("단기효과가 유의해도 장기효과는 대부분 없음. Top/Bottom 대비가 그 격차를 보여준다.")

    k = 5
    top = V.sort_values("net_afterglow", ascending=False).head(k)
    bot = V.sort_values("net_afterglow").head(k)
    tb  = pd.concat([bot.sort_values("net_afterglow"),
                     top.sort_values("net_afterglow")])

    # ① 처치 vs 반사실 Afterglow 비교 막대
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tb.festival_name, x=tb.afterglow, name="처치 Afterglow (개최 연)",
                         orientation="h", marker_color="#2E86DE",
                         text=tb.afterglow.round(2), textposition="outside"))
    fig.add_trace(go.Bar(y=tb.festival_name, x=tb.plac_afterglow, name="반사실 Afterglow (미개최 연)",
                         orientation="h", marker_color="#E59866",
                         text=tb.plac_afterglow.round(2), textposition="outside"))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="기준선 1.0")
    fig.add_hline(y=k - 0.5, line_dash="dot", line_color="black",
                  annotation_text="Top/Bottom 경계")
    fig.update_layout(barmode="group", height=560,
                      title="처치 vs 반사실 Afterglow — 파랑이 주황보다 길수록 순수 장기효과 큼",
                      xaxis_title="Afterglow (후3M ÷ 전3M)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # ② Net Lift + Net Afterglow 결합 비교
    fig2 = go.Figure()
    colors = ["#2E86DE" if v > 0 else "#C0392B" for v in tb.net_afterglow]
    fig2.add_trace(go.Bar(x=tb.festival_name, y=tb.net_afterglow, name="Net Afterglow",
                          marker_color=colors))
    fig2.add_trace(go.Scatter(x=tb.festival_name, y=tb.net_lift, name="Net Lift",
                              mode="markers", marker=dict(size=11, color="#8E44AD", symbol="diamond")))
    fig2.add_hline(y=0, line_color="black")
    fig2.update_layout(height=400,
                       title="Net Afterglow(막대) + Net Lift(점) 결합 비교",
                       yaxis_title="순효과", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top5 (장기효과 상위)**")
        st.dataframe(top.sort_values("net_afterglow", ascending=False)
                     [["festival_name","net_lift","net_afterglow"]].round(3),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Bottom5 (장기효과 하위)**")
        st.dataframe(bot.sort_values("net_afterglow")
                     [["festival_name","net_lift","net_afterglow"]].round(3),
                     use_container_width=True, hide_index=True)

# ── TAB 3: 선별 축제 ──────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"선별 축제 {len(SEL)}개 (기준B: Net Lift > 0 AND Net Afterglow > 0)")
    st.markdown("""
전체 44개 축제 중 **단기 효과(Net Lift > 0)와 장기 효과(Net Afterglow > 0)를 모두 충족**하는 축제.  
이 목록이 방문자·소비 데이터와의 종합분석 및 지역 특성 분석의 입력값이 된다.
""")

    # 선별 축제 산점도
    fig = px.scatter(SEL, x="net_lift", y="net_afterglow", text="festival_name",
                     size="vol_festival", color="net_afterglow",
                     color_continuous_scale="Blues",
                     labels={"net_lift":"Net Lift (단기)","net_afterglow":"Net Afterglow (장기)"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=520, title="선별 14개 축제 — 단기 vs 장기 순효과")
    st.plotly_chart(fig, use_container_width=True)

    # Net Afterglow 순위 막대
    s = SEL.sort_values("net_afterglow")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=s.festival_name, x=s.net_afterglow, name="Net Afterglow",
                          orientation="h", marker_color="#2E86DE",
                          text=s.net_afterglow.round(3), textposition="outside"))
    fig2.add_trace(go.Bar(y=s.festival_name, x=s.net_lift, name="Net Lift",
                          orientation="h", marker_color="#A9CCE3",
                          text=s.net_lift.round(3), textposition="inside"))
    fig2.update_layout(barmode="group", height=480,
                       title="선별 축제 Net Lift / Net Afterglow 비교",
                       xaxis_title="순효과 크기", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 선별 축제 목록")
    st.dataframe(
        SEL[["festival_name","sido","sigungu","n_editions",
             "lift","plac_lift","net_lift",
             "afterglow","plac_afterglow","net_afterglow"]].round(3),
        use_container_width=True, hide_index=True)

    # 비선별 축제도 참고용으로
    with st.expander("비선별 축제 목록 보기"):
        not_sel = V[~V.festival_name.isin(SEL.festival_name)].sort_values("net_afterglow", ascending=False)
        st.dataframe(not_sel[["festival_name","sido","sigungu",
                               "net_lift","net_afterglow"]].round(3),
                     use_container_width=True, hide_index=True)

# ── TAB 4: 개별 축제 상세 ────────────────────────────────────────────────────
with tab4:
    all_fests = V.sort_values("net_afterglow", ascending=False).festival_name.tolist()
    pick = st.selectbox("축제 선택", all_fests)
    row  = V[V.festival_name == pick].iloc[0]
    is_sel = pick in SEL.festival_name.values

    st.markdown(f"**{'✅ 선별 축제' if is_sel else '⬜ 비선별 축제'}** — "
                f"Net Lift {row.net_lift:+.3f} / Net Afterglow {row.net_afterglow:+.3f}")

    # 전후 프로파일
    prof = P[P.festival_name == pick].set_index("off")["idx100"].to_dict()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[OFFSET_LABEL[o] for o in range(-3, 4)],
        y=[prof.get(o) for o in range(-3, 4)],
        mode="lines+markers", line=dict(width=3, color="#2E86DE"), name="처치(개최 연)"))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="기준선 100")
    fig.update_layout(title=f"{pick} — 전후 검색량 프로파일 (기준선=100, 회차 평균)",
                      xaxis=dict(categoryorder="array", categoryarray=PERIOD_ORDER),
                      yaxis_title="정규화 검색량", height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Lift",      f"{row.net_lift:+.3f}",
              help=f"처치 {row.lift:.2f} − 반사실 {row.plac_lift:.2f}")
    c2.metric("Net Afterglow", f"{row.net_afterglow:+.3f}",
              help=f"처치 {row.afterglow:.2f} − 반사실 {row.plac_afterglow:.2f}")
    c3.metric("분석 회차",     f"{int(row.n_editions)}회")
    c4.metric("반사실 표본",   f"{int(row.n_placebo)}개")

# ── TAB 5: 방법론 · SQL ───────────────────────────────────────────────────────
with tab5:
    st.markdown(f"""
### 분석 목적
축제 개최가 지역의 **온라인 관심도를 단기적으로 끌어올리는지**, 나아가
**개최 이후에도 지속되어 실질적 지역 인지도 상승으로 이어지는지** 검증.

### 식별 전략 — 코로나 자연실험(이중차분, DiD)
분석 대상 축제 전부가 **2020·2021년 코로나로 미개최**였고 SNS 패널은 2020.01부터 존재.  
같은 시군구·같은 달의 2020/2021 값 = **'축제 없던 같은 시즌'** → 깨끗한 반사실.

| | 개최 연 | 미개최 연(2020·2021) |
|---|---|---|
| **관찰값** | 처치(treatment) | 반사실(placebo) |
| **지표** | Lift, Afterglow | 반사실 Lift, 반사실 Afterglow |
| **순효과** | **Net = 처치 − 반사실** | (계절성 상쇄) |

### 지표 정의 (±3개월 윈도우 유지)
- **Lift** = 축제月 ÷ 전후6개월(±3M, 당월 제외) 평균 → 단기 관심도 스파이크  
- **Afterglow** = 후3개월 평균 ÷ 전3개월 평균 → 장기 관심도 지속

### 분석 단위 & 통계 검정
- **분석 단위**: 회차 지표를 **축제별로 평균** → 독립표본 n={len(V)}개 (pseudoreplication 방지)
- **검정**: Wilcoxon signed-rank (처치 vs 반사실, paired, 비모수)
- **효과크기**: Net 중앙값 + 양(+) 비율 병기 (p값만으론 판단하지 않음)

### 결과 요약
- 단기효과 p = **{lp:.4f}** (유의) / Net Lift 중앙 **+{V.net_lift.median():.3f}**
- 장기효과 p = **{ap:.4f}** (비유의) / Net Afterglow 중앙 **{V.net_afterglow.median():+.3f}**
- **선별(기준B)**: Net Lift > 0 AND Net Afterglow > 0 → **{len(SEL)}/44개**

### 한계
- 반사실이 코로나 시기라 팬데믹 위축이 대조군에 일부 섞임 (효과 소폭 과대평가 가능)
- SNS 검색량은 시군구 단위 → 한 지역 다축제(논산·하동 등) 신호 혼입
- 검색량은 관심도의 대리지표 → 실제 방문·소비와는 별개
""")

    st.markdown("#### 핵심 분석 SQL — 처치/반사실 결합 및 축제 단위 집계")
    st.code(SQL_FESTIVAL, language="sql")
    st.markdown("#### 전후 프로파일 정규화 SQL")
    st.code(SQL_PROFILE, language="sql")

    sel_sql = """-- 선별 기준B: 단기(Net Lift) + 장기(Net Afterglow) 모두 양(+)
SELECT festival_name, sido, sigungu,
       (lift - plac_lift)           AS net_lift,
       (afterglow - plac_afterglow) AS net_afterglow
FROM <위 집계 결과>
WHERE (lift - plac_lift) > 0
  AND (afterglow - plac_afterglow) > 0
ORDER BY net_afterglow DESC;"""
    st.markdown("#### 선별 기준 SQL")
    st.code(sel_sql, language="sql")
