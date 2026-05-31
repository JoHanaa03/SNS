# -*- coding: utf-8 -*-
"""
축제가 온라인 관심도(SNS 검색량)에 미치는 영향 — 계절성 보정(이중차분) 분석 대시보드

핵심 개선:
  기존 "축제月 vs ±3개월" 비교는 축제효과와 계절수요가 섞임.
  → 코로나로 축제가 열리지 않은 2020·2021년의 '같은 달'을 반사실(counterfactual)로 사용.
    같은 ±3개월 윈도우·같은 지표를 적용해 계절성을 상쇄(difference-in-differences).
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
PLACEBO_YEARS = (2020, 2021)   # 코로나로 축제 미개최 → 자연 대조군

SQL_FEST = """
-- 분석 대상: 2020년 이후 개최된 모든 축제의 개최월(기준월 M)
SELECT sido, sigungu, festival_name, festival_year, year_month
FROM festival
WHERE period = 'FESTIVAL' AND festival_year >= 2020;
"""
SQL_SNS = """
-- 시군구 × 월 SNS 검색량 패널 (2020.01 ~ 2025.12)
SELECT sido, sigungu, year_month, search_volume
FROM sns_mention;
"""


@st.cache_data(show_spinner=False)
def load(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        fest = pd.read_sql(SQL_FEST, con)
        sns = pd.read_sql(SQL_SNS, con)
    finally:
        con.close()
    return fest, sns


def midx(year_month):
    return (year_month // 100) * 12 + (year_month % 100 - 1)


def build_panel(sns: pd.DataFrame) -> dict:
    sns = sns.copy()
    sns["m_idx"] = midx(sns["year_month"])
    return {(r.sido, r.sigungu, r.m_idx): r.search_volume for r in sns.itertuples()}


def window_metrics(panel, sido, sigungu, m_idx):
    """기준월 m_idx에 대해 ±3개월 윈도우 지표. 완전 윈도우가 아니면 None."""
    vf = panel.get((sido, sigungu, m_idx))
    bef = [panel.get((sido, sigungu, m_idx + o)) for o in (-3, -2, -1)]
    aft = [panel.get((sido, sigungu, m_idx + o)) for o in (1, 2, 3)]
    if vf is None or any(x is None for x in bef + aft):
        return None
    baseline = float(np.mean(bef + aft))
    return {
        "vol_festival": float(vf),
        "avg_before": float(np.mean(bef)),
        "avg_after": float(np.mean(aft)),
        "baseline": baseline,
        "lift": vf / baseline,
        "afterglow": float(np.mean(aft)) / float(np.mean(bef)),
        "profile": {o: panel.get((sido, sigungu, m_idx + o)) for o in range(-3, 4)},
    }


@st.cache_data(show_spinner=False)
def analyze(db_path: str):
    fest, sns = load(db_path)
    panel = build_panel(sns)
    fest = fest.copy()
    fest["moy"] = fest["year_month"] % 100

    edition_rows, fest_rows = [], []
    for (sido, sigungu, name), g in fest.groupby(["sido", "sigungu", "festival_name"]):
        # --- 처치(treatment): 실제 개최 회차 ---
        treat = []
        for _, r in g.iterrows():
            w = window_metrics(panel, sido, sigungu, midx(r.year_month))
            if w:
                w2 = {k: v for k, v in w.items() if k != "profile"}
                edition_rows.append({"sido": sido, "sigungu": sigungu,
                                     "festival_name": name,
                                     "festival_year": int(r.festival_year),
                                     "festival_ym": int(r.year_month),
                                     "kind": "처치(개최)", **w2})
                treat.append(w)
        if not treat:
            continue

        # --- 반사실(placebo): 같은 달, 축제 미개최 연도(2020/2021) ---
        plac = []
        for moy in sorted(set(g.moy)):
            for py in PLACEBO_YEARS:
                w = window_metrics(panel, sido, sigungu, py * 12 + (moy - 1))
                if w:
                    w2 = {k: v for k, v in w.items() if k != "profile"}
                    edition_rows.append({"sido": sido, "sigungu": sigungu,
                                         "festival_name": name,
                                         "festival_year": py,
                                         "festival_ym": py * 100 + moy,
                                         "kind": "반사실(미개최)", **w2})
                    plac.append(w)

        def m(lst, key):
            return float(np.mean([x[key] for x in lst])) if lst else np.nan

        # 처치 평균 프로파일(baseline=100 정규화) — 상세탭용
        prof = {o: [] for o in range(-3, 4)}
        for t in treat:
            bl = t["baseline"]
            for o, v in t["profile"].items():
                if v is not None and bl:
                    prof[o].append(v / bl * 100)
        prof_mean = {o: (float(np.mean(prof[o])) if prof[o] else np.nan)
                     for o in range(-3, 4)}

        fest_rows.append({
            "sido": sido, "sigungu": sigungu, "festival_name": name,
            "n_editions": int(g.festival_year.nunique()),
            "n_placebo": len(plac),
            "lift": m(treat, "lift"),
            "plac_lift": m(plac, "lift"),
            "afterglow": m(treat, "afterglow"),
            "plac_afterglow": m(plac, "afterglow"),
            "vol_festival": m(treat, "vol_festival"),
            "baseline": m(treat, "baseline"),
            "profile": prof_mean,
        })

    F = pd.DataFrame(fest_rows)
    F["net_lift"] = F.lift - F.plac_lift                 # 계절보정 단기 순효과
    F["net_afterglow"] = F.afterglow - F.plac_afterglow  # 계절보정 지속 순효과
    E = pd.DataFrame(edition_rows)
    return F, E


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
           "· ±3개월 윈도우 · 2020년 이후 개최 축제 전체")

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
    F, E = analyze(db_path)
except Exception as ex:
    st.error(f"DB 로딩/분석 실패: {ex}")
    st.stop()

if F.empty:
    st.warning("결합 결과가 비어 있습니다. 테이블/지역명 매칭을 확인하세요.")
    st.stop()

V = F.dropna(subset=["net_lift"]).copy()   # 반사실 확보 축제

ls, lp, ln = paired_test(V.lift, V.plac_lift)             # net lift
as_, ap, an = paired_test(V.afterglow, V.plac_afterglow)  # net afterglow
raw_s, raw_p, raw_n = paired_test(F.vol_festival, F.baseline)  # 계절 미보정 raw

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["핵심 결과", "계절보정 효과", "효과 유형 분류", "개별 축제", "방법론 · SQL"])

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
   즉 기존 분석의 상당 부분은 *축제 효과가 아니라 그 지역·그 계절의 자연 검색 증가*였다.

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

# ---------------- TAB 2 ----------------
with tab2:
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
    st.caption("대각선 위쪽 = 축제 연도가 미개최 연도보다 검색이 더 튐(순효과 +). "
               "아래쪽 = 계절 착시(축제 없어도 그 달에 원래 검색이 높음).")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("효과 유형 4분면: 단기 순효과 × 지속 순효과")
    Q = V.copy()
    fig = px.scatter(Q, x="net_lift", y="net_afterglow", text="festival_name",
                     size=Q.vol_festival, color="net_lift",
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

# ---------------- TAB 4 ----------------
with tab4:
    pick = st.selectbox("축제 선택", V.sort_values("net_lift", ascending=False).festival_name)
    row = F[F.festival_name == pick].iloc[0]

    sub = E[E.festival_name == pick]
    prof = row["profile"]
    figp = go.Figure()
    figp.add_trace(go.Scatter(
        x=[OFFSET_LABEL[o] for o in range(-3, 4)],
        y=[prof[o] for o in range(-3, 4)],
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
    c3.metric("개최 회차", f"{int(row.n_editions)}회")
    c4.metric("반사실 표본", f"{int(row.n_placebo)}개")

    st.markdown("##### 회차/반사실 원자료")
    st.dataframe(
        sub[["kind", "festival_year", "festival_ym", "vol_festival",
             "avg_before", "avg_after", "baseline", "lift", "afterglow"]].round(2),
        use_container_width=True, hide_index=True)

# ---------------- TAB 5 ----------------
with tab5:
    st.markdown(f"""
### 분석 목적
축제 개최가 해당 지역의 **온라인 관심도(SNS 검색량)** 를 *계절수요를 넘어서* 끌어올리는지,
또 그 효과가 **개최 이후에도 지속**되어 실질적 지역 인지도 상승으로 이어지는지 검증한다.

### 왜 기존 방식을 보정했나 (문제 정의)
대부분 매년 같은 시기에 열리는 **계절 축제**라, 단순히 "축제月 vs ±3개월"을 비교하면
*축제 효과* 와 *그 지역·그 계절의 자연 검색 수요* 가 분리되지 않는다.
실제로 보정 전 Lift 중앙값({F.lift.median():.2f}배)의 상당 부분이 계절성에서 비롯됨이 확인되었다.

### 식별 전략 — 코로나 자연실험 (이중차분, DiD)
분석 대상 축제 전부가 **2020·2021년에는 코로나로 미개최**였고, SNS 패널은 2020.01부터 존재한다.
따라서 같은 시군구·같은 달(month-of-year)의 2020/2021 값은 **'축제가 없었던 같은 시즌'** 이라는
깨끗한 반사실(counterfactual)이 된다.

- **처치(treatment)**: 개최 연도의 개최月 기준 ±3개월 윈도우 지표(Lift, Afterglow)
- **반사실(control)**: 같은 달·미개최 연도(2020/2021)의 동일 지표
- **순효과**: `Net = 처치 − 반사실` → 두 그룹에 공통인 계절성이 차감됨

### 지표 정의 (±3개월 윈도우 유지)
- **Lift** = 축제月 검색량 ÷ 전후 6개월(±3M, 당월 제외) 평균 → 단기 관심도 스파이크
- **Afterglow** = 후3개월 평균 ÷ 전3개월 평균 → 개최 후 관심 지속(지역 인지도 잔존)
- **Net Lift / Net Afterglow** = 위 지표의 처치 − 반사실

### 통계 기법 & 분석 단위
1. **분석 단위 분리**: 한 축제의 여러 회차는 독립이 아니므로(pseudoreplication),
   회차 지표를 **축제 단위로 평균**해 독립표본 n = {len(V)}개로 검정.
2. **Wilcoxon signed-rank (paired, 비모수)**: 처치 Lift vs 반사실 Lift를 짝지어 검정.
   소표본·비정규 분포에 강건. H₀: 처치 = 반사실(순효과 0).
3. **지속효과**도 동일하게 Afterglow에 대해 처치 vs 반사실 paired 검정.
4. **효과크기 병기**: p값뿐 아니라 Net 중앙값과 순효과 양(+) 축제 비율을 함께 보고.

### 결과 요약
- 단기효과: Net Lift 중앙 **{V.net_lift.median():+.2f}**, p = **{lp:.4f}** ({"유의" if lp<0.05 else "비유의"}),
  양(+) {int((V.net_lift>0).sum())}/{len(V)}.
- 지속효과: Net Afterglow 중앙 **{V.net_afterglow.median():+.2f}**, p = **{ap:.4f}** ({"유의" if ap<0.05 else "비유의"}).
- (참고) 계절 미보정 raw 검정 p = {raw_p:.4f} → 보정 시 효과가 보수적으로 줄어듦.

### 한계
- 반사실이 **코로나 시기(2020/2021)** 라, 팬데믹 자체의 이동·검색 위축이 대조군에 섞일 수 있음
  (효과를 다소 과대평가할 가능성). 비개최 연도가 더 있으면 보강 권장.
- SNS 검색량은 **시군구 단위** → 한 지역 다축제(논산·하동 등)는 신호 혼입.
- 검색량은 '관심도'의 대리지표이며 실제 방문·소비와는 별개 차원.
- 본 분석은 인과적 식별을 *근사* 하지만, 외부 사건(뉴스·바이럴) 완전 통제는 불가 → 강한 연관으로 해석.
""")
    st.markdown("#### 분석 대상 추출 SQL")
    st.code(SQL_FEST, language="sql")
    st.markdown("#### SNS 패널 SQL")
    st.code(SQL_SNS, language="sql")
    st.caption("±3개월 윈도우 구성, 처치/반사실 결합·집계·검정은 app.py의 pandas·scipy 로직에서 수행.")
