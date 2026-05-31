# -*- coding: utf-8 -*-
"""
축제가 온라인 관심도(SNS 검색량)에 미치는 영향 분석
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

SQL_MAIN = """
WITH fest_occ AS (
    SELECT sido, sigungu, festival_name,
           (year_month % 100)                         AS moy,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx,
           '처치' AS kind
    FROM festival
    WHERE period = 'FESTIVAL' AND festival_year >= 2020
),
placebo_occ AS (
    SELECT DISTINCT f.sido, f.sigungu, f.festival_name,
           f.moy, py.y*12 + (f.moy - 1) AS m_idx, '반사실' AS kind
    FROM fest_occ f
    CROSS JOIN (SELECT 2020 AS y UNION ALL SELECT 2021) py
),
joined AS (
    SELECT a.sido, a.sigungu, a.festival_name, a.kind,
           a.m_idx AS a_idx,
           (s.m_idx - a.m_idx) AS off, s.search_volume AS sv
    FROM (SELECT * FROM fest_occ UNION ALL SELECT * FROM placebo_occ) a
    JOIN (SELECT sido, sigungu, search_volume,
                 (year_month/100)*12 + (year_month%100 - 1) AS m_idx
          FROM sns_mention) s
      ON s.sido=a.sido AND s.sigungu=a.sigungu
     AND s.m_idx BETWEEN a.m_idx-3 AND a.m_idx+3
),
anchor_metrics AS (
    SELECT sido, sigungu, festival_name, kind, a_idx,
           1.0*MAX(CASE WHEN off=0 THEN sv END)
              /AVG(CASE WHEN off<>0 THEN sv END)                 AS lift,
           AVG(CASE WHEN off BETWEEN  1 AND 3 THEN sv END)
          /AVG(CASE WHEN off BETWEEN -3 AND -1 THEN sv END)      AS afterglow,
           AVG(CASE WHEN off=0 THEN sv END)                      AS vol_festival
    FROM joined
    GROUP BY sido, sigungu, festival_name, kind, a_idx
    HAVING COUNT(CASE WHEN off BETWEEN -3 AND -1 THEN 1 END) = 3
       AND COUNT(CASE WHEN off BETWEEN  1 AND  3 THEN 1 END) = 3
       AND MAX(CASE WHEN off=0 THEN 1 ELSE 0 END) = 1
)
SELECT sido, sigungu, festival_name,
       AVG(CASE WHEN kind='처치'  THEN lift        END) AS lift,
       AVG(CASE WHEN kind='반사실' THEN lift        END) AS plac_lift,
       AVG(CASE WHEN kind='처치'  THEN afterglow   END) AS afterglow,
       AVG(CASE WHEN kind='반사실' THEN afterglow   END) AS plac_afterglow,
       AVG(CASE WHEN kind='처치'  THEN vol_festival END) AS vol_festival
FROM anchor_metrics
GROUP BY sido, sigungu, festival_name;
"""

SQL_PROFILE = """
WITH fest_occ AS (
    SELECT sido, sigungu, festival_name, year_month AS anchor_ym,
           (year_month/100)*12 + (year_month%100 - 1) AS m_idx
    FROM festival WHERE period='FESTIVAL' AND festival_year >= 2020
),
w AS (
    SELECT f.festival_name, f.anchor_ym,
           (s.m_idx - f.m_idx) AS off, s.search_volume AS sv,
           AVG(CASE WHEN s.m_idx<>f.m_idx THEN s.search_volume END)
               OVER (PARTITION BY f.festival_name, f.anchor_ym) AS baseline,
           COUNT(CASE WHEN s.m_idx < f.m_idx THEN 1 END)
               OVER (PARTITION BY f.festival_name, f.anchor_ym) AS nb,
           COUNT(CASE WHEN s.m_idx > f.m_idx THEN 1 END)
               OVER (PARTITION BY f.festival_name, f.anchor_ym) AS na
    FROM fest_occ f
    JOIN (SELECT sido, sigungu, search_volume,
                 (year_month/100)*12+(year_month%100-1) AS m_idx
          FROM sns_mention) s
      ON s.sido=f.sido AND s.sigungu=f.sigungu
     AND s.m_idx BETWEEN f.m_idx-3 AND f.m_idx+3
)
SELECT festival_name, off, AVG(100.0*sv/baseline) AS idx100
FROM w WHERE nb=3 AND na=3
GROUP BY festival_name, off;
"""


@st.cache_data(show_spinner=False)
def load(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        F = pd.read_sql(SQL_MAIN, con)
        P = pd.read_sql(SQL_PROFILE, con)
    finally:
        con.close()
    F["net_lift"]      = F.lift - F.plac_lift
    F["net_afterglow"] = F.afterglow - F.plac_afterglow
    return F, P


def wilcoxon_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    try:
        w = stats.wilcoxon(a[m], b[m])
        return float(w.statistic), float(w.pvalue), int(m.sum())
    except Exception:
        return np.nan, np.nan, int(m.sum())


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("축제가 온라인 관심도에 미치는 영향")
st.caption("2020년 이후 개최 축제 · 이중차분(DiD) · ±3개월 윈도우")

with st.sidebar:
    st.header("설정")
    db_path = st.text_input("SQLite DB 경로", value="festival.db")
    st.divider()
    st.markdown("""
**선별 기준**  
Net Lift **> 0** (단기 ↑)  
Net Afterglow **> 0** (장기 ↑)
""")

try:
    F, P = load(db_path)
except Exception as ex:
    st.error(f"분석 실패: {ex}")
    st.stop()

# ── 수치 계산 ─────────────────────────────────────────────────────────────────
V   = F.dropna(subset=["net_lift","net_afterglow"]).copy()
SEL = V[(V.net_lift > 0) & (V.net_afterglow > 0)].sort_values("net_afterglow", ascending=False)

lp_w, lp_p, ln = wilcoxon_test(V.lift, V.plac_lift)
ap_w, ap_p, an = wilcoxon_test(V.afterglow, V.plac_afterglow)

tab1, tab2, tab3, tab4 = st.tabs(["분석 결과", "선별 축제", "개별 상세", "방법론 · SQL"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 축제", f"{len(V)}개")
    c2.metric("단기효과 p값", f"{lp_p:.4f}", "유의 ✓" if lp_p < 0.05 else "비유의")
    c3.metric("장기효과 p값", f"{ap_p:.4f}", "비유의 ✗" if ap_p >= 0.05 else "유의 ✓")
    c4.metric("온라인 관심도 상승 선별", f"{len(SEL)}/{len(V)}개")

    st.info(
        f"축제는 개최 당월 온라인 관심도를 **유의하게 끌어올리지만** (단기 p={lp_p:.4f}), "
        f"전체적으로 **장기 지속 근거는 없다** (장기 p={ap_p:.4f}) — 대부분 반짝 효과. "
        f"이 중 단기·장기 모두 계절보정 후 양(+)인 **{len(SEL)}개**를 '온라인 관심도 상승 지역'으로 선별한다.")

    # 4분면 산점도: Net Lift vs Net Afterglow
    V["구분"] = V.apply(
        lambda r: "선별" if r.net_lift > 0 and r.net_afterglow > 0 else "미선별", axis=1)

    fig = px.scatter(
        V, x="net_lift", y="net_afterglow",
        color="구분", color_discrete_map={"선별":"#2E86DE","미선별":"#D5D8DC"},
        text="festival_name",
        labels={"net_lift":"Net Lift (단기)","net_afterglow":"Net Afterglow (장기)","구분":""})
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.update_layout(height=520, showlegend=True,
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("우상단(파랑): 단기·장기 모두 양(+) → 선별 / 그 외(회색): 미선별")


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"선별 축제 {len(SEL)}개")
    st.caption("Net Afterglow(장기 순효과) 기준 내림차순")

    s = SEL.sort_values("net_afterglow")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=s.festival_name, x=s.net_afterglow,
        name="장기(Net Afterglow)", orientation="h",
        marker_color="#2E86DE",
        text=s.net_afterglow.round(2), textposition="outside"))
    fig.add_trace(go.Bar(
        y=s.festival_name, x=s.net_lift,
        name="단기(Net Lift)", orientation="h",
        marker_color="#85C1E9",
        text=s.net_lift.round(2), textposition="outside"))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="group", height=500,
                      xaxis_title="계절보정 순효과 크기",
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(t=40, r=80))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        SEL[["festival_name","sido","sigungu","net_lift","net_afterglow"]]
           .rename(columns={"festival_name":"축제명","sido":"시도","sigungu":"시군구",
                            "net_lift":"Net Lift","net_afterglow":"Net Afterglow"})
           .round(3).reset_index(drop=True),
        use_container_width=True, hide_index=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    fests = SEL.festival_name.tolist() + \
            V[~V.festival_name.isin(SEL.festival_name)]\
              .sort_values("net_afterglow", ascending=False).festival_name.tolist()
    pick = st.selectbox("축제 선택", fests)
    row  = V[V.festival_name == pick].iloc[0]
    is_sel = pick in SEL.festival_name.values

    badge = "✅ 선별" if is_sel else "⬜ 미선별"
    st.markdown(f"**{badge}** &nbsp;|&nbsp; Net Lift **{row.net_lift:+.2f}** &nbsp;/&nbsp; Net Afterglow **{row.net_afterglow:+.2f}**")

    prof = P[P.festival_name == pick].set_index("off")["idx100"].to_dict()
    xs   = [OFFSET_LABEL[o] for o in range(-3, 4)]
    ys   = [prof.get(o) for o in range(-3, 4)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=ys,
        marker_color=["#A9CCE3" if x != "FESTIVAL" else "#2E86DE" for x in xs],
        showlegend=False))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="기준선 100")
    fig.update_layout(
        title=f"{pick} — 개최 전후 온라인 검색량",
        xaxis=dict(categoryorder="array", categoryarray=PERIOD_ORDER),
        yaxis_title="상대 검색량 (기준월 평균=100)",
        height=400, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("FESTIVAL 월 막대(진파랑)가 기준선(100)을 크게 넘으면 단기 스파이크, AFTER 구간이 높으면 장기 지속.")


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
## 분석 목적
축제 개최가 지역 온라인 관심도를 **(1) 단기적으로 끌어올리는지**,
**(2) 이후에도 지속되어 실질적 지역 인지도 상승으로 이어지는지** 검증한다.

---

## 데이터
- **festival**: 2020년 이후 개최 축제별 period 행 (BEFORE_3M ~ AFTER_3M)
- **sns_mention**: 42개 시군 × 72개월 (2020.01–2025.12) SNS 검색량 패널

---

## 분석 설계 — 이중차분 (DiD)

**문제**: 대부분 매년 같은 달에 열리는 계절 축제라,
"축제月 vs ±3개월" 단순 비교는 축제 효과와 그 달의 자연 검색 수요(계절성)를 분리하지 못한다.

**해결 — 코로나 자연실험**: 44개 축제가 **2020·2021년 전부 미개최(코로나)** 였고
SNS 패널은 2020.01부터 존재한다. 같은 시군구·같은 달의 2020/2021 값이
**"축제 없던 같은 시즌"** 이라는 깨끗한 반사실(counterfactual)이 된다.

| 구분 | 정의 | 계산 |
|------|------|------|
| **처치** | 실제 개최 연도의 ±3개월 윈도우 지표 | Lift, Afterglow |
| **반사실** | 같은 달·미개최 연도(2020·2021)의 동일 지표 | 반사실 Lift, 반사실 Afterglow |
| **순효과** | 처치 − 반사실 | **Net Lift, Net Afterglow** |

---

## 지표 정의

| 지표 | 계산식 | 의미 |
|------|--------|------|
| **Lift** | 축제月 검색량 ÷ 전후 6개월 평균 | 단기 관심도 스파이크 |
| **Afterglow** | 후3개월 평균 ÷ 전3개월 평균 | 장기 관심도 지속 |
| **Net Lift** | 처치 Lift − 반사실 Lift | 계절보정 단기 순효과 |
| **Net Afterglow** | 처치 Afterglow − 반사실 Afterglow | 계절보정 장기 순효과 |

---

## 통계 기법

### 1. Pseudoreplication 방지 — 축제 단위 집계
같은 축제의 여러 회차는 서로 독립이 아니다(pseudoreplication).
처치/반사실 지표를 **축제별로 평균**해 독립표본 **n = {len(V)}개**로 검정한다.

### 2. Wilcoxon Signed-Rank Test (비모수 paired 검정)
- **H₀**: 처치 지표 = 반사실 지표 (순효과 = 0)
- **비모수 선택 이유**: n이 작고(44개) 검색량 분포의 정규성을 보장할 수 없어
  이상치에 강건한 순위 기반 검정을 사용
- **Paired 선택 이유**: 같은 축제·같은 달에서 처치와 반사실을 짝지어 비교하므로
  개체 내 통제(within-unit control)가 가능해 검정력이 높아짐

| | 통계량 W | p값 | 결론 |
|--|--|--|--|
| **단기(Net Lift)** | {lp_w:.1f} | **{lp_p:.4f}** | {"유의 — H₀ 기각" if lp_p<0.05 else "비유의"} |
| **장기(Net Afterglow)** | {ap_w:.1f} | **{ap_p:.4f}** | {"유의 — H₀ 기각" if ap_p<0.05 else "비유의 — H₀ 기각 불가"} |

### 3. 효과크기 병기
p값은 표본 크기에 의존하므로 **Net 중앙값**과 **양(+) 비율**을 함께 보고한다.

| | 중앙값 | 양(+) 비율 |
|--|--|--|
| Net Lift | +{V.net_lift.median():.3f} | {int((V.net_lift>0).sum())}/{len(V)} |
| Net Afterglow | {V.net_afterglow.median():+.3f} | {int((V.net_afterglow>0).sum())}/{len(V)} |

### 4. 선별 기준 (기준B)
Net Lift > 0 **AND** Net Afterglow > 0 → **{len(SEL)}/{len(V)}개** 선별

---

## 한계
1. **반사실 = 코로나 시기** — 팬데믹 이동 위축이 대조군에 일부 섞여 순효과 소폭 과대평가 가능
2. **시군구 단위 SNS** — 한 지역 다축제(논산·하동 등) 신호 혼입
3. **대리지표** — 검색량 ≠ 실제 방문·소비
4. **연관 ≠ 인과** — 외부 사건(바이럴 등) 완전 통제 불가

---
""")
    st.markdown("### 핵심 SQL")
    st.code(SQL_MAIN, language="sql")
    st.markdown("### 프로파일 SQL")
    st.code(SQL_PROFILE, language="sql")
