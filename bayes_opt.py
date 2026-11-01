from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood


TORCH_DTYPE = torch.double
DEFAULT_BETA = 2.0
MIN_NOISE_VARIANCE = 1e-12
RESULTS_DIR = Path("result")
GRAPH_DIR = Path("BO_graph")
PLOT_GRID_SIZE = 80
BLACK_BOX_RNG = np.random.default_rng(20260409)
PYTHON = ".venv/bin/python"
RUNS_DIR = Path("runs")
LOG_DIR = Path("log_temp")
VIDEO_DIR = Path("videos")
NUM_ENVS = 4096
MAX_ITERATIONS = 2500
VIDEO_LENGTH = 480  # 8 s
TIMESTAMP_SUFFIX_PATTERN = re.compile(r"_(\d{14})$")


@dataclass(frozen=True)
class Observation:
    number: int | None
    time: str | None
    x1: float
    x2: float
    y: float
    y_var: float


def black_box_function(x1: float, x2: float, current_step: int) -> tuple[float, float]:
    """
    MyAnt (Genesis): 目的関数は強化学習 (rsl_rl PPO) の最大平均報酬。
    x1 = 脚 (leg) の長さ, x2 = 足 (foot) の長さ。地球重力 (-9.81 m/s^2)。

    Returns:
        tuple[float, float]:
            y: -MaxMeanReward (最小化問題にするため符号反転)
            y_var: 学習ログ最後の100回分の Mean reward の分散 (train_Yvar として渡す)
    """
    leg_length = x1  # default 0.2828427255153656
    foot_length = x2  # default 0.5656854510307312

    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M%S")
    run_name = f"step{current_step:02d}_{time_str}"
    run_dir = RUNS_DIR / run_name
    log_path = LOG_DIR / f"log_{time_str}_{current_step}.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # run (Isaac Lab 版の train.py 呼び出しに相当)
    os.system(
        f"{PYTHON} train.py --leg_length {leg_length} --foot_length {foot_length} "
        f"--num_envs {NUM_ENVS} --max_iterations {MAX_ITERATIONS} --log_dir {run_dir} > {log_path} 2>&1"
    )

    # save movie (Isaac Lab 版の play.py --video に相当)
    video_path = VIDEO_DIR / f"step{current_step:02d}_leg{leg_length:.3f}_foot{foot_length:.3f}.mp4"
    os.system(
        f"{PYTHON} play.py --leg_length {leg_length} --foot_length {foot_length} "
        f"--checkpoint {run_dir / 'model_final.pt'} --video {video_path} --video_length {VIDEO_LENGTH} "
        f">> {log_path} 2>&1"
    )

    MeanRewards = []
    with open(log_path, "r") as f:
        for line in f:
            if line.find("Mean reward") >= 0:
                MeanRewards.append(float(line.split()[2]))

    if not MeanRewards:
        raise RuntimeError(f"training failed, see {log_path}")

    MaxMeanReward = max(MeanRewards)
    # 最後の100イテレーション分の平均報酬の分散
    # (MyAnt_train 版は MeanRewards[:-100] で「最後の100件以外」になっていたため修正)
    MeanRewards_var = np.var(MeanRewards[-100:])

    y = MaxMeanReward * -1  # 最大化問題を最小化問題に変換するために -1 を掛ける
    y_var = MeanRewards_var
    return y, y_var



class NegativeLowerConfidenceBound(AnalyticAcquisitionFunction):
    """
    Negative LCB for minimization.

    BoTorch maximizes acquisition functions, so we return `-(mu - beta * sigma)`.
    Maximizing this value is equivalent to minimizing the LCB.
    """

    def __init__(self, model: SingleTaskGP, beta: float = DEFAULT_BETA) -> None:
        super().__init__(model=model)
        self.beta = float(beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(MIN_NOISE_VARIANCE).squeeze(-1).squeeze(-1)
        sigma = variance.sqrt()
        lcb = mean - self.beta * sigma
        return -lcb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BoTorch による 2 変数ブラックボックス関数のベイズ最適化"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    suggest = subparsers.add_parser("suggest-next", help="次に評価すべき x1, x2 を提案する")
    add_common_arguments(suggest)
    suggest.add_argument(
        "--acquisition",
        choices=("lcb", "ucb"),
        default="lcb",
        help="最小化問題では ucb を指定しても -y に対する UCB として LCB と等価になります。",
    )
    suggest.add_argument("--beta", type=float, default=DEFAULT_BETA, help="LCB/UCB の係数")
    suggest.add_argument(
        "--num-restarts", type=int, default=10, help="獲得関数最適化の再スタート回数"
    )
    suggest.add_argument(
        "--raw-samples", type=int, default=128, help="獲得関数最適化の初期サンプル数"
    )

    register = subparsers.add_parser("register", help="評価済みデータを observations.csv に追加する")
    add_observation_file_argument(register)
    register.add_argument("--x1", type=float, required=True, help="入力 x1")
    register.add_argument("--x2", type=float, required=True, help="入力 x2")
    register.add_argument("--y", type=float, required=True, help="目的値 y")
    register.add_argument(
        "--y-var",
        type=float,
        required=True,
        dest="y_var",
        help="観測ごとの分散値を指定してください。",
    )

    best = subparsers.add_parser("best", help="観測済みデータから現在の最良点を表示する")
    add_observation_file_argument(best)

    run = subparsers.add_parser(
        "run-loop",
        help="black_box_function() を使って初期点生成から逐次最適化まで自動実行する",
    )
    add_common_arguments(run)
    run.add_argument("--beta", type=float, default=DEFAULT_BETA, help="LCB の係数")
    run.add_argument("--max-evals", type=int, required=True, help="総評価回数")
    run.add_argument(
        "--num-restarts", type=int, default=10, help="獲得関数最適化の再スタート回数"
    )
    run.add_argument(
        "--raw-samples", type=int, default=128, help="獲得関数最適化の初期サンプル数"
    )
    run.add_argument(
        "--prior-observations",
        type=Path,
        nargs="*",
        default=[],
        help="事前データとして使う過去の観測CSV。複数指定可。",
    )
    run.add_argument(
        "--initial-design",
        choices=("random", "specified"),
        default="random",
        help="GP 構築前の初期点をランダム生成するか、指定点を使うか。",
    )
    run.add_argument(
        "--initial-point",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X1", "X2"),
        help="initial-design=specified のときに使う初期点。--n-initial 回数分を指定してください。",
    )

    return parser.parse_args()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    add_observation_file_argument(parser)
    parser.add_argument(
        "--lower-bounds",
        type=float,
        nargs=2,
        metavar=("X1_MIN", "X2_MIN"),
        required=True,
        help="x1, x2 の下限値",
    )
    parser.add_argument(
        "--upper-bounds",
        type=float,
        nargs=2,
        metavar=("X1_MAX", "X2_MAX"),
        required=True,
        help="x1, x2 の上限値",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=4,
        help="GP 構築前に Sobol で提案する初期点数",
    )
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")


def add_observation_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="観測データ CSV の保存先",
    )


def validate_bounds(lower_bounds: Iterable[float], upper_bounds: Iterable[float]) -> torch.Tensor:
    lower = torch.tensor(list(lower_bounds), dtype=TORCH_DTYPE)
    upper = torch.tensor(list(upper_bounds), dtype=TORCH_DTYPE)
    if lower.numel() != 2 or upper.numel() != 2:
        raise ValueError("lower_bounds と upper_bounds は x1, x2 の 2 要素で指定してください。")
    if not torch.all(upper > lower):
        raise ValueError("各次元の upper_bounds は lower_bounds より大きくしてください。")
    return torch.stack([lower, upper], dim=0)


def load_observations(path: Path) -> list[Observation]:
    if not path.exists():
        return []

    observations: list[Observation] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        legacy_fields = {"x1", "x2", "y", "y_var"}
        current_fields = {"Number", "Time", "x1", "x2", "y", "y_var"}
        if reader.fieldnames is None:
            raise ValueError("CSV ヘッダーを読み取れませんでした。")
        actual_fields = set(reader.fieldnames)
        if actual_fields not in (legacy_fields, current_fields):
            raise ValueError("CSV のヘッダーは x1,x2,y,y_var または Number,Time,x1,x2,y,y_var にしてください。")
        for row in reader:
            observations.append(
                Observation(
                    number=int(row["Number"]) if "Number" in row and row["Number"] else None,
                    time=row["Time"] if "Time" in row and row["Time"] else None,
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    y=float(row["y"]),
                    y_var=max(float(row["y_var"]), MIN_NOISE_VARIANCE),
                )
            )
    return observations


def append_observation(path: Path, observation: Observation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Number", "Time", "x1", "x2", "y", "y_var"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "Number": observation.number,
                "Time": observation.time,
                "x1": observation.x1,
                "x2": observation.x2,
                "y": observation.y,
                "y_var": max(observation.y_var, MIN_NOISE_VARIANCE),
            }
        )


def format_timestamp(dt: datetime) -> str:
    return dt.astimezone().isoformat(timespec="milliseconds")


def format_filename_timestamp(dt: datetime) -> str:
    return dt.astimezone().strftime("_%Y%m%d%H%M%S")


def resolve_observation_path(path: Path | None, *, now: datetime, for_write: bool) -> Path:
    if path is None:
        if not for_write:
            raise ValueError("--observations を指定してください。")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return RESULTS_DIR / f"observations{format_filename_timestamp(now)}.csv"

    resolved = path
    if for_write and not resolved.is_absolute():
        if resolved.parent == Path("."):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            resolved = RESULTS_DIR / resolved.name
        else:
            resolved.parent.mkdir(parents=True, exist_ok=True)

    if for_write and not resolved.exists():
        if resolved.suffix.lower() != ".csv":
            resolved = resolved.with_suffix(".csv")
        timestamp_suffix = format_filename_timestamp(now)
        if not resolved.stem.endswith(timestamp_suffix):
            resolved = resolved.with_name(f"{resolved.stem}{timestamp_suffix}{resolved.suffix}")
        resolved.parent.mkdir(parents=True, exist_ok=True)

    return resolved


def resolve_existing_observation_path(path: Path) -> Path:
    if path.exists():
        return path

    if not path.is_absolute() and path.parent == Path("."):
        candidate = RESULTS_DIR / path
        if candidate.exists():
            return candidate

        if path.suffix.lower() == ".csv":
            matches = sorted(RESULTS_DIR.glob(path.name))
        else:
            matches = sorted(RESULTS_DIR.glob(f"{path.name}_*.csv"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(f"観測CSVが見つかりません: {path}")


def get_next_observation_number(observations: list[Observation]) -> int:
    numbered = [obs.number for obs in observations if obs.number is not None]
    if numbered:
        return max(numbered) + 1
    return len(observations) + 1


def renumber_observations(observations: list[Observation], start: int = 1) -> list[Observation]:
    return [
        Observation(
            number=index,
            time=obs.time,
            x1=obs.x1,
            x2=obs.x2,
            y=obs.y,
            y_var=obs.y_var,
        )
        for index, obs in enumerate(observations, start=start)
    ]


def load_prior_observations(paths: list[Path]) -> list[Observation]:
    prior_observations: list[Observation] = []
    for path in paths:
        resolved_path = resolve_existing_observation_path(path)
        prior_observations.extend(load_observations(resolved_path))
    return prior_observations


def seed_prior_observations_sequentially(
    observations_path: Path,
    observations: list[Observation],
    prior_observations: list[Observation],
    bounds: torch.Tensor,
    n_initial: int,
    beta: float,
    seed: int,
    num_restarts: int,
    raw_samples: int,
    initial_design: str,
    initial_points: list[tuple[float, float]],
) -> list[Observation]:
    seeded_observations = observations.copy()
    renumbered_priors = renumber_observations(prior_observations, start=len(seeded_observations) + 1)

    for prior_observation in renumbered_priors:
        append_observation(observations_path, prior_observation)
        seeded_observations.append(prior_observation)
        next_step_candidate, _ = suggest_candidate(
            observations=seeded_observations,
            bounds=bounds,
            n_initial=n_initial,
            beta=beta,
            seed=seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            initial_design=initial_design,
            initial_points=initial_points,
        )
        graph_path = plot_optimization_state(
            observations=seeded_observations,
            bounds=bounds,
            beta=beta,
            next_candidate=next_step_candidate,
            step=prior_observation.number if prior_observation.number is not None else len(seeded_observations),
            now=datetime.now(),
            observations_path=observations_path,
        )
        print(
            f"seeded_prior_observation: Number={prior_observation.number}, "
            f"x1={prior_observation.x1}, x2={prior_observation.x2}, saved_plot={graph_path}"
        )

    return seeded_observations


def validate_initial_points(
    initial_points: list[list[float]] | list[tuple[float, float]],
    bounds: torch.Tensor,
    n_initial: int,
    initial_design: str,
) -> list[tuple[float, float]]:
    parsed_points = [(float(point[0]), float(point[1])) for point in initial_points]
    if initial_design == "specified":
        if len(parsed_points) != n_initial:
            raise ValueError(
                "initial-design=specified のときは --initial-point を --n-initial 回数分だけ指定してください。"
            )
        for x1, x2 in parsed_points:
            if not (bounds[0, 0].item() <= x1 <= bounds[1, 0].item()):
                raise ValueError(f"指定した初期点 x1={x1} が bounds 外です。")
            if not (bounds[0, 1].item() <= x2 <= bounds[1, 1].item()):
                raise ValueError(f"指定した初期点 x2={x2} が bounds 外です。")
    return parsed_points


def generate_initial_candidate(bounds: torch.Tensor, seed: int) -> torch.Tensor:
    samples_scaled = draw_sobol_samples(bounds=get_unit_bounds(), n=1, q=1, seed=seed)
    return unnormalize_inputs(samples_scaled.squeeze(0).squeeze(0), bounds=bounds)


def build_training_tensors(observations: list[Observation]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[obs.x1, obs.x2] for obs in observations], dtype=TORCH_DTYPE)
    train_Y = torch.tensor([[obs.y] for obs in observations], dtype=TORCH_DTYPE)
    train_Yvar = torch.tensor(
        [[max(obs.y_var, MIN_NOISE_VARIANCE)] for obs in observations],
        dtype=TORCH_DTYPE,
    )
    return train_X, train_Y, train_Yvar


def get_unit_bounds() -> torch.Tensor:
    return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=TORCH_DTYPE)


def normalize_inputs(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    lower = bounds[0]
    upper = bounds[1]
    return (X - lower) / (upper - lower)


def unnormalize_inputs(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    lower = bounds[0]
    upper = bounds[1]
    return lower + X * (upper - lower)


def fit_model(observations: list[Observation], bounds: torch.Tensor) -> SingleTaskGP:
    train_X, train_Y, train_Yvar = build_training_tensors(observations)
    train_X_scaled = normalize_inputs(train_X, bounds=bounds)
    model = SingleTaskGP(
        train_X=train_X_scaled,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def build_plot_grid(
    bounds: torch.Tensor, grid_size: int = PLOT_GRID_SIZE
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    x1_values = np.linspace(bounds[0, 0].item(), bounds[1, 0].item(), grid_size)
    x2_values = np.linspace(bounds[0, 1].item(), bounds[1, 1].item(), grid_size)
    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    grid_points = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
    grid_tensor = torch.tensor(grid_points, dtype=TORCH_DTYPE)
    grid_tensor_scaled = normalize_inputs(grid_tensor, bounds=bounds)
    return x1_mesh, x2_mesh, grid_tensor, grid_tensor_scaled


def create_plot_path(step: int, now: datetime) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPH_DIR / f"acquisition_{now.strftime('%Y%m%d%H%M%S')}_{step}.png"


def extract_run_timestamp(observations_path: Path) -> str:
    match = TIMESTAMP_SUFFIX_PATTERN.search(observations_path.stem)
    if match is None:
        raise ValueError(
            f"CSV ファイル名から日時を抽出できませんでした: {observations_path.name}"
        )
    return match.group(1)


def create_plot_path_for_run(observations_path: Path, step: int, now: datetime) -> Path:
    run_timestamp = extract_run_timestamp(observations_path)
    plot_dir = GRAPH_DIR / run_timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f"acquisition_{now.strftime('%Y%m%d%H%M%S')}_{step}.png"


def plot_optimization_state(
    observations: list[Observation],
    bounds: torch.Tensor,
    beta: float,
    next_candidate: torch.Tensor,
    step: int,
    now: datetime,
    observations_path: Path,
) -> Path:
    if not observations:
        raise ValueError("プロットには少なくとも1件の観測が必要です。")

    model = fit_model(observations=observations, bounds=bounds)
    acquisition = NegativeLowerConfidenceBound(model=model, beta=beta)
    x1_mesh, x2_mesh, _, grid_tensor_scaled = build_plot_grid(bounds=bounds)

    with torch.no_grad():
        posterior = model.posterior(grid_tensor_scaled)
        posterior_mean = posterior.mean.reshape(x1_mesh.shape).detach().cpu().numpy()
        posterior_std = posterior.variance.clamp_min(MIN_NOISE_VARIANCE).sqrt().reshape(x1_mesh.shape).detach().cpu().numpy()
        acquisition_values = (
            acquisition(grid_tensor_scaled.unsqueeze(1)).reshape(x1_mesh.shape).detach().cpu().numpy()
        )

    observed_x1 = [obs.x1 for obs in observations]
    observed_x2 = [obs.x2 for obs in observations]
    next_x1 = float(next_candidate[0].item())
    next_x2 = float(next_candidate[1].item())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plot_specs = [
        ("Posterior mean", posterior_mean),
        ("Posterior sd.", posterior_std),
        ("Acquisition function", acquisition_values),
    ]

    for axis, (title, values) in zip(axes, plot_specs):
        contour = axis.contourf(x1_mesh, x2_mesh, values, levels=30, cmap="viridis")
        fig.colorbar(contour, ax=axis)
        axis.scatter(observed_x1, observed_x2, c="tomato", s=36, edgecolors="white", linewidths=0.6)
        axis.set_title(title)
        axis.set_xlabel("X1")
        axis.set_ylabel("X2")
        axis.set_xlim(bounds[0, 0].item(), bounds[1, 0].item())
        axis.set_ylim(bounds[0, 1].item(), bounds[1, 1].item())

    axes[2].scatter(
        [next_x1],
        [next_x2],
        c="magenta",
        s=500,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
    )

    output_path = create_plot_path_for_run(observations_path=observations_path, step=step, now=now)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def suggest_candidate(
    observations: list[Observation],
    bounds: torch.Tensor,
    n_initial: int,
    beta: float,
    seed: int,
    num_restarts: int,
    raw_samples: int,
    initial_design: str = "random",
    initial_points: list[tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, str]:
    initial_points = initial_points or []
    if len(observations) < n_initial:
        if initial_design == "specified":
            x1, x2 = initial_points[len(observations)]
            candidate = torch.tensor([x1, x2], dtype=TORCH_DTYPE)
            return candidate, f"Specified initial design ({len(observations) + 1}/{n_initial})"
        candidate = generate_initial_candidate(bounds=bounds, seed=seed + len(observations))
        return candidate, f"Sobol initial design ({len(observations) + 1}/{n_initial})"

    torch.manual_seed(seed)
    model = fit_model(observations=observations, bounds=bounds)
    acquisition = NegativeLowerConfidenceBound(model=model, beta=beta)
    candidate_scaled, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=get_unit_bounds(),
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    candidate = unnormalize_inputs(candidate_scaled.squeeze(0), bounds=bounds)
    return candidate, f"LCB(beta={beta})"


def print_candidate(candidate: torch.Tensor, strategy: str) -> None:
    print(f"strategy: {strategy}")
    print(f"x1: {candidate[0].item():.10f}")
    print(f"x2: {candidate[1].item():.10f}")


def handle_suggest_next(args: argparse.Namespace) -> None:
    bounds = validate_bounds(args.lower_bounds, args.upper_bounds)
    if args.observations is None:
        raise ValueError("suggest-next では --observations で既存CSVを指定してください。")
    observations_path = resolve_existing_observation_path(args.observations)
    observations = load_observations(observations_path)
    candidate, strategy = suggest_candidate(
        observations=observations,
        bounds=bounds,
        n_initial=args.n_initial,
        beta=args.beta,
        seed=args.seed,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )
    if args.acquisition == "ucb":
        strategy += " (minimization problem: UCB on -y is equivalent to LCB on y)"
    print_candidate(candidate, strategy)


def handle_register(args: argparse.Namespace) -> None:
    now = datetime.now()
    observations_path = resolve_observation_path(args.observations, now=now, for_write=True)
    observations = load_observations(observations_path)
    observation = Observation(
        number=get_next_observation_number(observations),
        time=format_timestamp(now),
        x1=args.x1,
        x2=args.x2,
        y=args.y,
        y_var=args.y_var,
    )
    append_observation(observations_path, observation)
    print(
        f"registered: Number={observation.number}, Time={observation.time}, "
        f"x1={args.x1}, x2={args.x2}, y={args.y}, y_var={args.y_var}, "
        f"path={observations_path}"
    )


def handle_best(args: argparse.Namespace) -> None:
    if args.observations is None:
        raise ValueError("best では --observations で既存CSVを指定してください。")
    observations_path = resolve_existing_observation_path(args.observations)
    observations = load_observations(observations_path)
    if not observations:
        raise ValueError("観測データがありません。まず register でデータを追加してください。")
    best = min(observations, key=lambda obs: obs.y)
    print(f"best_y: {best.y}")
    print(f"x1: {best.x1}")
    print(f"x2: {best.x2}")
    print(f"y_var: {best.y_var}")


def handle_run_loop(args: argparse.Namespace) -> None:
    now = datetime.now()
    bounds = validate_bounds(args.lower_bounds, args.upper_bounds)
    initial_points = validate_initial_points(
        initial_points=args.initial_point,
        bounds=bounds,
        n_initial=args.n_initial,
        initial_design=args.initial_design,
    )
    observations_path = resolve_observation_path(args.observations, now=now, for_write=True)
    observations = load_observations(observations_path)
    if not observations and args.prior_observations:
        prior_observations = load_prior_observations(args.prior_observations)
        observations = seed_prior_observations_sequentially(
            observations_path=observations_path,
            observations=observations,
            prior_observations=prior_observations,
            bounds=bounds,
            n_initial=args.n_initial,
            beta=args.beta,
            seed=args.seed,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            initial_design=args.initial_design,
            initial_points=initial_points,
        )
        print(f"seeded_prior_observations: count={len(prior_observations)}")
    program_start_time = now
    initial_observation_count = len(observations)

    while len(observations) < initial_observation_count + args.max_evals:
        step_start_time = datetime.now()
        candidate, strategy = suggest_candidate(
            observations=observations,
            bounds=bounds,
            n_initial=args.n_initial,
            beta=args.beta,
            seed=args.seed,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            initial_design=args.initial_design,
            initial_points=initial_points,
        )
        x1 = float(candidate[0].item())
        x2 = float(candidate[1].item())
        current_step = len(observations) - initial_observation_count + 1
        print(f"[{current_step}/{args.max_evals}] evaluating by {strategy}: x1={x1}, x2={x2}")
        timestamp = program_start_time if current_step <= args.n_initial else step_start_time
        y, y_var = black_box_function(x1, x2, current_step)
        observation = Observation(
            number=get_next_observation_number(observations),
            time=format_timestamp(timestamp),
            x1=x1,
            x2=x2,
            y=float(y),
            y_var=float(y_var),
        )
        append_observation(observations_path, observation)
        observations.append(observation)
        print(f"recorded: Number={observation.number}, Time={observation.time}, y={y}, y_var={y_var}")
        next_step_candidate, _ = suggest_candidate(
            observations=observations,
            bounds=bounds,
            n_initial=args.n_initial,
            beta=args.beta,
            seed=args.seed,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            initial_design=args.initial_design,
            initial_points=initial_points,
        )
        graph_path = plot_optimization_state(
            observations=observations,
            bounds=bounds,
            beta=args.beta,
            next_candidate=next_step_candidate,
            step=observation.number if observation.number is not None else len(observations),
            now=datetime.now(),
            observations_path=observations_path,
        )
        print(f"saved_plot: {graph_path}")

    best = min(observations, key=lambda obs: obs.y)
    print("optimization finished")
    print(f"observations_path: {observations_path}")
    print(f"best_y: {best.y}")
    print(f"x1: {best.x1}")
    print(f"x2: {best.x2}")


def main() -> None:
    args = parse_args()
    if args.command == "suggest-next":
        handle_suggest_next(args)
        return
    if args.command == "register":
        handle_register(args)
        return
    if args.command == "best":
        handle_best(args)
        return
    if args.command == "run-loop":
        handle_run_loop(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
