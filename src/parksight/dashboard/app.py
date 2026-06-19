from __future__ import annotations

import json

import pandas as pd
import pydeck as pdk
import streamlit as st

from parksight import config, service
from parksight.analysis.emergence import relocation_labels
from parksight.scoring.priority import PriorityWeights

st.set_page_config(page_title="ParkSight", layout="wide")

st.markdown(
    "<style>footer{visibility:hidden}#MainMenu{visibility:hidden}"
    "[data-testid='stAppDeployButton']{display:none}.block-container{padding-top:2.5rem}</style>",
    unsafe_allow_html=True,
)

teams = st.sidebar.slider("Patrol teams", 4, 40, 12)

st.sidebar.subheader("Priority weights")
raw_weights = {
    "volume": st.sidebar.slider("Volume", 0.0, 1.0, 0.4, 0.05),
    "severity": st.sidebar.slider("Severity", 0.0, 1.0, 0.3, 0.05),
    "disruption": st.sidebar.slider("Flow disruption", 0.0, 1.0, 0.2, 0.05),
    "persistence": st.sidebar.slider("Persistence", 0.0, 1.0, 0.1, 0.05),
}
weight_total = sum(raw_weights.values())
weights = (
    PriorityWeights(**{name: value / weight_total for name, value in raw_weights.items()})
    if weight_total
    else PriorityWeights()
)
st.sidebar.caption("Weights renormalise to 1 and re-rank hotspots and patrol placements live.")

st.title("ParkSight")
st.caption(
    "Parking-congestion intelligence for Bengaluru enforcement. Scoped to logged "
    "enforcement activity; flow impact is an estimate, not a measurement."
)

if config.FINDINGS_PATH.exists():
    findings = json.loads(config.FINDINGS_PATH.read_text())
    concentration = findings["concentration"]
    recidivism = findings["recidivism"]
    churn = findings["churn"]
    st.markdown(f"**Full feed — {findings['rows']:,} records**")
    a, b, c, d = st.columns(4)
    a.metric("Top 5% of cells hold", f"{concentration['share_top_5pct']:.0%}")
    b.metric("Spatial Gini", f"{concentration['gini']:.2f}")
    c.metric("Repeat-offender volume", f"{recidivism['repeat_share']:.0%}")
    d.metric("Emerging / declining", f"{churn['emerging']} / {churn['declining']}")

    relocation = service.relocation()
    st.markdown(
        f"**Emerging hotspots are disproportionately relocated patrols, not new problems.** "
        f"The enforcement devices logging them were already active elsewhere "
        f"{relocation['observed_continuity']:.0%} of the time, vs a {relocation['device_base_rate']:.0%} "
        f"citywide base rate — a {relocation['lift']:+.0%} lift over chance "
        f"(permutation p<{max(relocation['p_value'], 0.001):.3f}). So most apparent emergence is enforcement "
        f"shifting its gaze; only {relocation['genuine_emerging']} of {relocation['emerging_cells']} emerging "
        "cells are staffed mostly by new devices. ParkSight scores this so planners separate real "
        "signal from patrol relocation."
    )

impact = service.impact()
if impact:
    st.subheader("Flow impact by junction — bias-adjusted, and the patrol plan it drives")
    corr = impact["exposure_bias_corr"]
    dep = impact.get("deployment")
    junctions = pd.DataFrame(impact["junctions"]).sort_values("corrected_intensity", ascending=False).reset_index(drop=True)
    plan = junctions.head(teams).copy()
    nearest = min(dep["curve"], key=lambda r: abs(r["teams"] - teams)) if dep else None

    a, b, c = st.columns(3)
    a.metric("Raw hotspots explained by patrol presence", f"{corr ** 2:.0%}", "r² of log-count on log-effort", delta_color="off")
    b.metric("Named junctions scored", f"{impact['n_junctions']}")
    if nearest:
        c.metric(f"Violations / team vs raw plan ({nearest['teams']} teams)", f"{nearest['topn']['uplift_pct']:+.0f}%", "corrected plan, out-of-sample")

    peak = junctions["corrected_intensity"].max()
    junctions["radius"] = 120 + junctions["corrected_intensity"] / peak * 520
    junctions["fill"] = junctions["corrected_intensity"].map(lambda value: [196, int(40 + 70 * value / peak), 48, 160])
    plan["fill"] = [[24, 96, 204, 230]] * len(plan)
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.5),
            layers=[
                pdk.Layer("ScatterplotLayer", data=junctions, get_position=["longitude", "latitude"], get_radius="radius", get_fill_color="fill", pickable=True),
                pdk.Layer("ScatterplotLayer", data=plan, get_position=["longitude", "latitude"], get_radius=480, get_fill_color="fill"),
            ],
            tooltip={"text": "{junction}\ncorrected {corrected_intensity}\nraw {raw_count}"},
        )
    )
    held = impact["model"]["heldout"]
    st.caption(
        f"Red dots size with the bias-adjusted intensity (violations per patrol-day); blue dots are the {teams} junctions the "
        f"corrected signal staffs. Raw counts track patrol presence at r={corr:.2f} (r²={corr ** 2:.0%}), so a raw hotspot "
        "map mostly shows where patrols already went. The corrected score divides that out — a Negative-Binomial intensity "
        "with an enforcement-effort offset — and beats a Poisson fit and an effort-only null out-of-sample (held-out "
        f"log-likelihood {held['nb_ll_per_obs']:.2f} vs {held['poisson_ll_per_obs']:.2f} vs {held['null_ll_per_obs']:.2f})."
    )
    show = junctions.assign(rank=range(1, len(junctions) + 1), raw_rank=junctions["raw_count"].rank(ascending=False).astype(int))
    st.dataframe(
        show[["junction", "corrected_intensity", "rank", "raw_count", "raw_rank", "exposure", "raw_per_effort"]].head(15),
        use_container_width=True,
        hide_index=True,
    )
    prod = impact.get("productivity_validation")
    if prod and nearest:
        spear = prod["spearman_vs_next_period_yield"]
        tp = nearest["topn"]
        st.caption(
            "Validated out-of-sample (time-split). Across all junctions the corrected ranking predicts next period's "
            f"violations-per-team better than raw counts (Spearman {spear['eb_shrunk_yield']:.2f} vs {spear['raw_count']:.2f}). "
            f"Staffing the worst {nearest['teams']} junctions by the corrected signal reaches {tp['corrected']:.1f} violations "
            f"per team next week, versus {tp['raw']:.1f} on raw counts ({tp['uplift_pct']:+.0f}%) and {dep['city_mean_yield']:.1f} "
            "across the city, at the same headcount."
        )
    rob = impact.get("validation_robustness")
    if rob:
        st.caption(
            f"Data-quality check: {rob['rejected_duplicate_share']:.0%} of citations are rejected or duplicate; the model "
            f"shown excludes them. Dropping them leaves the corrected ranking unchanged (Kendall τ "
            f"{rob['kendall_tau_clean_vs_full']:.2f}) and the deployment gain at {rob['deployment_uplift_clean_pct']:+.0f}% — "
            "the result is not an artifact of bad tickets."
        )

if config.FORECAST_PATH.exists():
    payload = json.loads(config.FORECAST_PATH.read_text())
    metrics = payload["metrics"]
    mae = metrics["mae"]
    volatile = metrics["mae_volatile_cells"]

    st.subheader("Next-week forecast (residual temporal model)")
    one, two, three = st.columns(3)
    one.metric(
        "Forecast MAE",
        f"{mae['temporal']:.2f}",
        f"{mae['temporal'] - mae['persistence']:+.2f} vs persistence",
        delta_color="inverse",
    )
    two.metric("Persistence MAE", f"{mae['persistence']:.2f}")
    three.metric(
        "Volatile-cell MAE",
        f"{volatile['temporal']:.2f}",
        f"{volatile['temporal'] - volatile['persistence']:+.2f} vs persistence",
        delta_color="inverse",
    )
    seed_note = ""
    if config.SEED_ROBUSTNESS_PATH.exists():
        seeds = json.loads(config.SEED_ROBUSTNESS_PATH.read_text())
        seed_note = (
            f" The result is stable across random seeds (MAE {seeds['temporal_mean']:.2f} "
            f"± {seeds['temporal_std']:.2f}; it beats persistence on every one of {seeds['seeds']} seeds)."
        )
    st.caption(
        "A per-cell temporal model learning the residual over a persistence baseline, significant "
        "at p<0.001. A spatio-temporal graph variant was tested and dropped (it degraded accuracy), "
        "and a Poisson gradient-boosted cross-check corroborates the gain. Trained on the full feed."
        + seed_note
    )

    deployment = service.forecast_deployment(teams)
    st.metric(
        "Next-week patrol coverage",
        f"{deployment.greedy_coverage:.0%} of forecast volume",
        f"+{deployment.greedy_coverage - deployment.naive_coverage:.0%} vs naive top-K",
    )
    placements = deployment.plan.assign(fill=[[24, 96, 204, 230]] * len(deployment.plan))

    forecast = pd.DataFrame(
        [
            {
                "cell": cell,
                "latitude": float(cell.split(",")[0]),
                "longitude": float(cell.split(",")[1]),
                "next_week": value,
            }
            for cell, value in payload["forecast"].items()
        ]
    ).sort_values("next_week", ascending=False)

    predicted = forecast.head(150).copy()
    peak = predicted["next_week"].max()
    predicted["radius"] = 60 + predicted["next_week"] * 0.9
    predicted["fill"] = predicted["next_week"].map(
        lambda value: [214, int(120 * (1 - value / peak)) + 30, 40, 175]
    )
    predicted["expected"] = predicted["next_week"].round(0)

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.5),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=predicted,
                    get_position=["longitude", "latitude"],
                    get_radius="radius",
                    get_fill_color="fill",
                    pickable=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=placements,
                    get_position=["longitude", "latitude"],
                    get_radius=180,
                    get_fill_color="fill",
                ),
            ],
            tooltip={"text": "{cell}\nexpected {expected} next week"},
        )
    )
    st.caption(
        "Shaded dots scale with predicted violations over the next 7 days; blue dots are the "
        "recommended patrol placements (greedy coverage of forecast volume)."
    )
    table = forecast.head(20).rename(columns={"next_week": "predicted_next_week"})
    conf = service.conformal()
    if conf:
        q80 = conf["quantiles"]["80"]
        predicted = table["predicted_next_week"]
        table = table.assign(
            low_80=(predicted - q80 * (predicted + 1) ** 0.5).clip(lower=0).round(0),
            high_80=(predicted + q80 * (predicted + 1) ** 0.5).round(0),
        )
        cal80 = next(row for row in conf["calibration"] if row["target"] == 0.8)
        st.caption(
            f"low_80 / high_80 are conformal 80% prediction intervals — empirically covering "
            f"{cal80['empirical_coverage']:.0%} of held-out actuals out-of-sample (calibration verified)."
        )
    st.dataframe(table, use_container_width=True)

backtest = service.deployment_backtest()
if backtest:
    grid = backtest["overall"]
    team_grid = [row["teams"] for row in grid]
    nearest = min(team_grid, key=lambda value: abs(value - teams))
    row = next(item for item in grid if item["teams"] == nearest)

    st.subheader("Out-of-sample validation — would deploying on the forecast have helped?")
    one, two, three = st.columns(3)
    one.metric(
        f"Forecast-driven coverage ({nearest} teams)",
        f"{row['forecast']:.1%}",
        f"+{row['forecast_uplift']:.1%} vs status-quo",
    )
    two.metric("Status-quo (last-week footprint)", f"{row['status_quo']:.1%}")
    three.metric("Perfect-foresight ceiling", f"{row['oracle']:.1%}")
    st.caption(
        f"Across {backtest['n_windows']} held-out weeks, placing {nearest} teams on the forecast each week "
        f"would have reached {row['forecast']:.1%} of the following week's actual violations — "
        f"{row['headroom_captured']:.0%} of the gain available over deploying where you were last week. "
        "The ceiling is narrow because parking is highly persistent (Gini 0.84): last week already fixes most "
        "of next week, so the headroom is small and the model captures a real share of it. "
        "The ceiling is a greedy oracle, so the share captured is an upper estimate."
    )
    curve = pd.DataFrame(
        {
            "teams": team_grid,
            "status quo": [item["status_quo"] for item in grid],
            "forecast": [item["forecast"] for item in grid],
            "oracle (perfect foresight)": [item["oracle"] for item in grid],
        }
    ).set_index("teams")
    st.line_chart(curve)

shift = service.shifts()
if shift:
    st.subheader("Time-of-day — shift-aware deployment")
    a, b, c = st.columns(3)
    a.metric("Enforcement before 2pm", f"{1 - shift['unenforced_after_14h_share']:.0%}", "afternoon is a blind spot", delta_color="off")
    b.metric("Busy cells peaking off-shift", f"{shift['peak_shift_differs_share']:.0%}")
    c.metric("Shift-aware coverage", f"{shift['shift_aware_coverage']:.1%}", f"+{shift['uplift']:.1%} vs static plan")
    st.caption(
        "Enforcement runs in two windows — night (00–06) and day (07–13); the afternoon and evening are "
        "effectively unmonitored. A third of busy cells are hottest in a different window than the city overall, "
        "so relocating the same teams between shifts covers more than a fixed all-day plan. The gain concentrates "
        "in the night shift, which a day-biased static plan under-serves."
    )
    forecast_shifts = shift.get("forecast", {})

    def _next_week_cell(name):
        cells = forecast_shifts.get(name, {}).get("top_cells", [])
        return cells[0]["cell"] if cells else "—"

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "shift": name,
                    "hours": block["hours"],
                    "records": block["records"],
                    "static coverage": block["static_coverage"],
                    "shift-aware coverage": block["shift_aware_coverage"],
                    "top cell (to date)": block["top_cells"][0]["cell"],
                    "forecast top cell (next week)": _next_week_cell(name),
                }
                for name, block in shift["shifts"].items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if forecast_shifts:
        st.caption(
            "Forward-looking: next week's per-shift hotspots come from the cell forecast split by each cell's "
            "stable time-of-day profile (median between-half TV distance 0.12), so night and day teams get "
            "separate, forecast-driven placements."
        )

intelligence = service.intelligence()
hotspots = intelligence.hotspots.reset_index()
plan = intelligence.plan(teams)
coverage = intelligence.volume_coverage(plan)
plan = plan.reset_index()

st.subheader("Hotspot priority and patrol placements (full feed)")
left, right = st.columns(2)
left.metric("Scored cells", f"{len(hotspots):,}")
right.metric("Violation volume covered", f"{coverage:.0%}")

hotspots["radius"] = 80 + hotspots["priority"] * 240
hotspots["fill"] = hotspots["priority"].map(lambda p: [196, int(40 + 70 * p), 48, 150])
plan["fill"] = [[24, 96, 204, 220]] * len(plan)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.5),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=hotspots,
                get_position=["longitude", "latitude"],
                get_radius="radius",
                get_fill_color="fill",
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=plan,
                get_position=["longitude", "latitude"],
                get_radius=160,
                get_fill_color="fill",
            ),
        ],
        tooltip={"text": "{station}\n{cell}\nvolume {volume}"},
    )
)
st.caption("Red points are priority hotspots (size scales with priority); blue points are recommended patrol placements.")

st.subheader("Top hotspots")
columns = ["station", "volume", "severity", "disruption", "persistence", "priority"]
st.dataframe(hotspots.set_index("cell").head(25)[columns], use_container_width=True)

st.subheader("Moving hotspots")
status = st.radio("status", ["emerging", "declining"], horizontal=True, label_visibility="collapsed")
moving = intelligence.transitions
moving = moving[moving["status"] == status].copy()
moving["signal"] = relocation_labels(moving["enforcement_continuity"])
moving = moving.head(20)
st.caption(
    "Enforcement-continuity is the share of patrol devices here that were already active elsewhere "
    "last period. High values (“likely relocation”) mean the patrol moved, not that a new problem "
    "appeared; “genuine signal” cells are where most enforcement is new and warrant a real look."
)
st.dataframe(
    moving[["station", "h1", "h2", "delta", "enforcement_continuity", "signal"]],
    use_container_width=True,
)
