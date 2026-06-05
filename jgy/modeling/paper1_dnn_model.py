from __future__ import annotations

"""논문1 방식의 DNN 산불 피해면적 예측 모델 코드.

논문1: "기상 데이터를 이용한 데이터 마이닝 기반의 산불 예측 모델"

논문에서 가장 성능이 좋다고 설명한 모델은 기상 변수 5개만 사용하는 DNN
회귀 모델입니다. 이 파일은 그 구조를 우리 프로젝트에서 실행할 수 있도록
구현한 코드입니다.

중요한 점
--------
이 파일은 논문 PDF 안에 들어 있던 "학습 완료 모델"을 가져온 것이 아닙니다.
논문에는 모델 구조와 실험 방법이 설명되어 있고, 실제 가중치 파일은 없습니다.
그래서 여기서는 논문 설명을 바탕으로 DNN 모델을 새로 구현했습니다.

입력 변수
--------
논문 M 셋업, meteorological features:

1. avg_temp
2. min_temp
3. max_temp
4. max_wind_speed
5. avg_wind

타깃 변수
--------
논문은 산불 피해면적을 그대로 예측하지 않고 아래처럼 로그 변환합니다.

    y = ln(피해면적 + 1)

예측 결과를 다시 실제 ha 단위로 볼 때는 아래처럼 되돌립니다.

    피해면적 = exp(y) - 1

현재 우리 데이터의 문제
--------------------
data/강원도_날씨데이터/강원도날씨_통합_시간단위.csv 에서 만든
paper1_dnn_gangwon_weather_input_from_integrated_time.csv 는 입력 변수는
준비되어 있습니다. 하지만 강원도 2020~2021 산불 데이터의 피해면적(ha)이
비어 있어서 지금 당장은 논문처럼 지도학습을 완료하기 어렵습니다.

따라서 이 파일의 역할은 다음과 같습니다.

1. 피해면적 타깃이 있는 CSV가 생기면 DNN을 학습한다.
2. 학습된 가중치를 data/modeling/paper1_dnn_weights.npz 로 저장한다.
3. 저장된 모델로 강원도 날씨 입력 파일에 대해 피해면적 예측값을 만든다.

실행 예시
--------
학습:

    python jgy/modeling/paper1_dnn_model.py train --train-csv data/modeling/train.csv

예측:

    python jgy/modeling/paper1_dnn_model.py predict

입력만 점검:

    python jgy/modeling/paper1_dnn_model.py check

의존성
------
TensorFlow, Keras, scikit-learn이 현재 환경에 없어서 NumPy만으로 작은 DNN을
구현했습니다. 논문 구조를 재현하기 위한 실험용 코드이며, 최종 제출 모델로
쓰려면 타깃 데이터 확보 후 성능 검증이 필요합니다.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "modeling"
DEFAULT_INPUT = OUT_DIR / "paper1_dnn_gangwon_weather_input_from_integrated_time.csv"
DEFAULT_WEIGHTS = OUT_DIR / "paper1_dnn_weights.npz"
DEFAULT_PREDICTIONS = OUT_DIR / "paper1_dnn_gangwon_weather_predictions.csv"

FEATURES = ["avg_temp", "min_temp", "max_temp", "max_wind_speed", "avg_wind"]
TARGET_CANDIDATES = ["burned_area", "피해면적(ha)", "damage_area_ha"]


@dataclass
class TrainResult:
    train_mae_log: float
    valid_mae_log: float
    valid_mae_ha: float
    valid_rmse_ha: float


class Paper1DNN:
    """Small fully connected DNN for log burned-area regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        learning_rate: float = 1e-3,
        optimizer: str = "rmsprop",
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.rng = np.random.default_rng(random_state)

        dims = [input_dim] + [hidden_units] * hidden_layers + [1]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        self.opt_state = {
            "w": [np.zeros_like(w) for w in self.weights],
            "b": [np.zeros_like(b) for b in self.biases],
            "vw": [np.zeros_like(w) for w in self.weights],
            "vb": [np.zeros_like(b) for b in self.biases],
            "t": 0,
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        pre_activations = []
        h = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = h @ weight + bias
            h = np.maximum(z, 0)
            pre_activations.append(z)
            activations.append(h)
        y_pred = h @ self.weights[-1] + self.biases[-1]
        pre_activations.append(y_pred)
        activations.append(y_pred)
        return y_pred, activations, pre_activations

    def predict_log(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].ravel()

    def predict_area(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(self.predict_log(x)).clip(min=0)

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred, activations, pre_activations = self.forward(x)
        y = y.reshape(-1, 1)
        loss = np.mean((y_pred - y) ** 2)

        grad = 2 * (y_pred - y) / len(x)
        grad_w = []
        grad_b = []

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w.insert(0, a_prev.T @ grad)
            grad_b.insert(0, grad.sum(axis=0, keepdims=True))
            if layer_idx > 0:
                grad = grad @ self.weights[layer_idx].T
                grad = grad * (pre_activations[layer_idx - 1] > 0)

        self.apply_gradients(grad_w, grad_b)
        return float(loss)

    def apply_gradients(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        eps = 1e-8
        self.opt_state["t"] += 1

        for i, (gw, gb) in enumerate(zip(grad_w, grad_b)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * gw
                self.biases[i] -= self.learning_rate * gb
            elif self.optimizer == "adagrad":
                self.opt_state["w"][i] += gw**2
                self.opt_state["b"][i] += gb**2
                self.weights[i] -= self.learning_rate * gw / (np.sqrt(self.opt_state["w"][i]) + eps)
                self.biases[i] -= self.learning_rate * gb / (np.sqrt(self.opt_state["b"][i]) + eps)
            else:
                decay = 0.9
                self.opt_state["w"][i] = decay * self.opt_state["w"][i] + (1 - decay) * gw**2
                self.opt_state["b"][i] = decay * self.opt_state["b"][i] + (1 - decay) * gb**2
                self.weights[i] -= self.learning_rate * gw / (np.sqrt(self.opt_state["w"][i]) + eps)
                self.biases[i] -= self.learning_rate * gb / (np.sqrt(self.opt_state["b"][i]) + eps)

    def save(self, path: Path, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        payload = {
            "x_mean": x_mean,
            "x_std": x_std,
            "feature_names": np.array(FEATURES),
            "num_layers": np.array([len(self.weights)]),
        }
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weight_{i}"] = weight
            payload[f"bias_{i}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> tuple["Paper1DNN", np.ndarray, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"][0])
        weights = [data[f"weight_{i}"] for i in range(num_layers)]
        biases = [data[f"bias_{i}"] for i in range(num_layers)]

        model = cls(input_dim=weights[0].shape[0], hidden_layers=1, hidden_units=1)
        model.weights = weights
        model.biases = biases
        return model, data["x_mean"], data["x_std"]


def find_target_column(df: pd.DataFrame) -> str:
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"타깃 컬럼이 없습니다. 가능한 이름: {TARGET_CANDIDATES}")


def load_features(df: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"필수 입력 피처가 없습니다: {missing}")
    return df[FEATURES].astype(float).to_numpy()


def standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1
    return (x - x_mean) / x_std, x_mean, x_std


def train(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    target_col = find_target_column(df)
    df = df.dropna(subset=FEATURES + [target_col]).copy()
    if df.empty:
        raise ValueError("학습 가능한 행이 없습니다. 입력 피처와 피해면적 타깃의 결측을 확인하세요.")

    x_raw = load_features(df)
    y_area = df[target_col].astype(float).clip(lower=0).to_numpy()
    y = np.log1p(y_area)
    x, x_mean, x_std = standardize_train(x_raw)

    rng = np.random.default_rng(args.random_state)
    indices = rng.permutation(len(x))
    split = int(len(indices) * (1 - args.valid_size))
    train_idx, valid_idx = indices[:split], indices[split:]

    model = Paper1DNN(
        input_dim=len(FEATURES),
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        random_state=args.random_state,
    )

    for epoch in range(1, args.epochs + 1):
        shuffled = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            batch_idx = shuffled[start : start + args.batch_size]
            losses.append(model.train_batch(x[batch_idx], y[batch_idx]))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            valid_pred = model.predict_log(x[valid_idx])
            valid_mae = np.mean(np.abs(valid_pred - y[valid_idx]))
            print(f"epoch={epoch} train_mse={np.mean(losses):.4f} valid_mae_log={valid_mae:.4f}")

    result = evaluate(model, x[train_idx], y[train_idx], x[valid_idx], y[valid_idx])
    model.save(args.weights, x_mean, x_std)

    print(f"saved={args.weights}")
    print(f"train_mae_log={result.train_mae_log:.4f}")
    print(f"valid_mae_log={result.valid_mae_log:.4f}")
    print(f"valid_mae_ha={result.valid_mae_ha:.4f}")
    print(f"valid_rmse_ha={result.valid_rmse_ha:.4f}")


def evaluate(
    model: Paper1DNN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> TrainResult:
    train_pred = model.predict_log(x_train)
    valid_pred = model.predict_log(x_valid)
    valid_area = np.expm1(y_valid).clip(min=0)
    pred_area = np.expm1(valid_pred).clip(min=0)
    return TrainResult(
        train_mae_log=float(np.mean(np.abs(train_pred - y_train))),
        valid_mae_log=float(np.mean(np.abs(valid_pred - y_valid))),
        valid_mae_ha=float(np.mean(np.abs(pred_area - valid_area))),
        valid_rmse_ha=float(np.sqrt(np.mean((pred_area - valid_area) ** 2))),
    )


def predict(args: argparse.Namespace) -> None:
    model, x_mean, x_std = Paper1DNN.load(args.weights)
    df = pd.read_csv(args.predict_csv, encoding="utf-8-sig")
    x = (load_features(df) - x_mean) / x_std
    df["paper1_pred_log_burned_area"] = model.predict_log(x)
    df["paper1_pred_burned_area_ha"] = model.predict_area(x)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"created={args.output_csv}")
    print(f"rows={len(df)}")


def check(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.predict_csv, encoding="utf-8-sig")
    print(f"file={args.predict_csv}")
    print(f"shape={df.shape}")
    print(f"required_features={FEATURES}")
    print(f"missing_features={[column for column in FEATURES if column not in df.columns]}")
    if "date" in df.columns:
        print(f"date_range={df['date'].min()}..{df['date'].max()}")
    print(df[FEATURES].describe().round(3).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper1 DNN burned-area model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-csv", type=Path, required=True)
    train_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--valid-size", type=float, default=0.2)
    train_parser.add_argument("--hidden-layers", type=int, default=3)
    train_parser.add_argument("--hidden-units", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--optimizer", choices=["sgd", "adagrad", "rmsprop"], default="rmsprop")
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--print-every", type=int, default=20)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--predict-csv", type=Path, default=DEFAULT_INPUT)
    predict_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    predict_parser.add_argument("--output-csv", type=Path, default=DEFAULT_PREDICTIONS)
    predict_parser.set_defaults(func=predict)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--predict-csv", type=Path, default=DEFAULT_INPUT)
    check_parser.set_defaults(func=check)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
