import argparse
import subprocess
import sys
import tempfile
from itertools import chain
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

RECALL_TOP_KS = [1, 5, 10]
RECALL_MAX_K = 1000
SCATTER_COLOR = "#ffa15a"
VIOLIN_COLOR = "#203864"
REF_TABLE = {"=": "=", "ガ": "NOM", "ヲ": "ACC", "ニ": "DAT", "デ": "INS-LOC", "ノ": "Bridging"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_names", type=str, nargs="+", help="Experiment name (directory name under args.root-dir)")
    parser.add_argument("--root-dir", type=Path, default="result/mmref")
    parser.add_argument(
        "--id-file", type=Path, nargs="+", default=[Path("data/id/test.id")], help="Paths to scenario id file"
    )
    parser.add_argument("--rel-types", type=str, nargs="+", default=["="])
    args = parser.parse_args()
    scenario_ids: list[str] = list(chain.from_iterable(path.read_text().splitlines() for path in args.id_file))
    for exp_name in args.exp_names:
        with tempfile.TemporaryDirectory() as out_dir:
            command = (
                f"{sys.executable} src/evaluation.py -p {args.root_dir / exp_name} --scenario-ids {' '.join(scenario_ids)}"
                f" --recall-topk {' '.join(map(str, [*RECALL_TOP_KS, RECALL_MAX_K]))} --th 0.0 --raw-result-csv {out_dir}/raw_result.csv"
                f" --show-average-confidences"
            )
            subprocess.run(command.split(), cwd=Path.cwd(), check=True)
            df_raw_result = pl.read_csv(f"{out_dir}/raw_result.csv")
        output_dir = Path("data") / "confidence_distribution"
        output_dir.mkdir(exist_ok=True)
        for rel_type in args.rel_types:
            visualize(df_raw_result, output_dir / f"{exp_name}_{rel_type}.pdf", rel_type)


def visualize(comparison_table: pl.DataFrame, output_file: Path, rel_type: str) -> None:
    comparison_table = (
        comparison_table.filter(pl.col("rel_type") == rel_type)
        .filter(pl.col("class_name") != "")
        .drop(["scenario_id", "image_id", "sid", "base_phrase_index", "rel_type", "instance_id_or_pred_idx"])
    )
    data = []
    for row in comparison_table.to_dicts():
        if row["recall_pos@10"] == 0:
            continue
        for idx, recall_topk in enumerate(RECALL_TOP_KS):
            data.append(
                {"id": idx, "rank": f"top-{recall_topk}", "confidence": row[f"recall_avg_conf@top{recall_topk}"]}
            )
            data.append(
                {
                    "id": idx + len(RECALL_TOP_KS),
                    "rank": f"bottom-{recall_topk}",
                    "confidence": row[f"recall_avg_conf@bottom{recall_topk}"],
                }
            )
        data.append({"id": RECALL_MAX_K, "rank": "ALL", "confidence": row[f"recall_avg_conf@top{RECALL_MAX_K}"]})

    df_confidence = pl.DataFrame(data)
    df_confidence = df_confidence.sort("id")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_confidence["confidence"],
            y=df_confidence["rank"],
            name="Confidence Score Averages",
            mode="markers",
            marker=dict(size=6, color=SCATTER_COLOR),
            opacity=0.75,
        )
    )

    fig.add_trace(
        go.Violin(
            x=df_confidence["confidence"],
            y=df_confidence["rank"],
            box_visible=False,
            meanline_visible=False,
            points=False,
            side="negative",
            width=1.75,
            bandwidth=0.05,
            orientation="h",
            showlegend=False,
            line=dict(color=VIOLIN_COLOR),
        )
    )

    # Plot median
    df_confidence_median = df_confidence.group_by("rank").agg(pl.median("confidence"))
    fig.add_trace(
        go.Scatter(
            x=df_confidence_median["confidence"],
            y=df_confidence_median["rank"],
            mode="markers",
            marker=dict(color="red", symbol="star"),
            name="Median",
        )
    )

    fig.update_layout(
        barmode="overlay",
        legend=dict(
            yanchor="top",
            xanchor="left",
            y=0.99,
            x=0.01,
            orientation="h",
        ),
        xaxis=dict(
            title=f"Confidence Score for {REF_TABLE[rel_type]}",
        ),
        # https://plotly.com/python/reference/layout/yaxis/
        yaxis=dict(title="Recall@10 Correct Predictions by Confidence", autorange="reversed"),
    )

    # https://github.com/plotly/plotly.py/issues/3469
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig.write_image(output_file)


if __name__ == "__main__":
    main()
