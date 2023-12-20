import io
import subprocess
import sys
from pathlib import Path

import plotly.express as px
import polars as pl
from plotly.colors import qualitative

RECALL_TOP_KS = (1, 5, 10, -1)
MAX_CLASSES_TO_SHOW = 20


def main() -> None:
    project_root = Path.cwd()
    interpreter = sys.executable
    scenario_ids: list[str] = (
        project_root.joinpath("data/id/test.id").read_text().splitlines()
        + project_root.joinpath("data/id/valid.id").read_text().splitlines()
    )
    exp_name = sys.argv[1]
    output = subprocess.run(
        (
            f"{interpreter} src/evaluation.py -p result/mmref/{exp_name} --scenario-ids {' '.join(scenario_ids)}"
            f" --recall-topk {' '.join(map(str, RECALL_TOP_KS))} --th 0.0 --eval-modes class --format csv"
        ).split(),
        capture_output=True,
        cwd=project_root,
        check=True,
        text=True,
    )
    class_score_table = pl.read_csv(io.StringIO(output.stdout))
    visualize(class_score_table)


def visualize(class_score_table: pl.DataFrame) -> None:
    for recall_topk in RECALL_TOP_KS:
        metric_suffix = f"@{recall_topk}" if recall_topk >= 0 else ""
        class_score_table = class_score_table.rename({f"recall_pos{metric_suffix}": f"Recall{metric_suffix}"})
    class_score_table = (
        class_score_table.rename({"recall_total": "Total"})
        .filter(pl.col("class_name") != "")
        .sort(pl.col("Total"), descending=True)
    )[:MAX_CLASSES_TO_SHOW]
    fig = px.bar(
        class_score_table,
        x="class_name",
        y=["Total", "Recall", "Recall@10", "Recall@5", "Recall@1"],
        # https://plotly.com/python/discrete-color/
        color_discrete_sequence=[
            qualitative.Plotly[3],
            qualitative.Plotly[1],
            qualitative.Plotly[4],
            qualitative.Plotly[7],
            qualitative.Plotly[5],
        ],
    )
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            title="Class Name",
            titlefont=dict(size=25),
            tickfont=dict(size=20),
            tickangle=55,
            categoryorder="array",
            categoryarray=class_score_table.sort(pl.col("Total"), descending=True)["class_name"].to_list(),
        ),
        yaxis=dict(
            title="Object-bounding Boxes",
            titlefont=dict(size=25),
            tickfont=dict(size=20),
        ),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.97,
            font=dict(size=18),
            title=dict(text=""),
        ),
    )
    # https://github.com/plotly/plotly.py/issues/3469
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig.write_image("recall_distribution.pdf")
    # fig.show()


if __name__ == "__main__":
    main()