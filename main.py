from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pydicom


POSITIVE_CLASS = "Pneumothorax"
NEGATIVE_CLASS = "No Pneumothorax"
DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_HIST_BINS = 256
Strategy = Literal["nearest", "class_mean", "class_top3"]
PreprocessMode = Literal["raw", "equalize", "clahe"]


@dataclass(frozen=True)
class HistMethod:
    name: str
    flag: int
    higher_is_better: bool


@dataclass(frozen=True)
class MethodConfig:
    image_size: tuple[int, int]
    hist_bins: int
    preprocess: PreprocessMode
    strategy: Strategy


HIST_METHODS: tuple[HistMethod, ...] = (
    HistMethod("CV_COMP_CORREL (Correlation)", cv2.HISTCMP_CORREL, True),
    HistMethod("CV_COMP_CHISQR (Chi-Square)", cv2.HISTCMP_CHISQR, False),
    HistMethod("CV_COMP_INTERSECT (Intersection)", cv2.HISTCMP_INTERSECT, True),
    HistMethod("CV_COMP_BHATTACHARYYA (Bhattacharyya distance)", cv2.HISTCMP_BHATTACHARYYA, False),
)


METHOD_CONFIGS: dict[str, MethodConfig] = {
    "CV_COMP_CORREL (Correlation)": MethodConfig(
        image_size=(96, 96),
        hist_bins=32,
        preprocess="clahe",
        strategy="class_mean",
    ),
    "CV_COMP_CHISQR (Chi-Square)": MethodConfig(
        image_size=(128, 128),
        hist_bins=64,
        preprocess="clahe",
        strategy="class_mean",
    ),
    "CV_COMP_INTERSECT (Intersection)": MethodConfig(
        image_size=(64, 64),
        hist_bins=256,
        preprocess="clahe",
        strategy="nearest",
    ),
    "CV_COMP_BHATTACHARYYA (Bhattacharyya distance)": MethodConfig(
        image_size=(64, 64),
        hist_bins=32,
        preprocess="raw",
        strategy="class_mean",
    ),
}


@dataclass
class Sample:
    path: Path
    label: str
    histogram: np.ndarray


def parse_list_file(file_path: Path, label: str) -> list[tuple[Path, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    entries: list[tuple[Path, str]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        entries.append((Path(text), label))
    return entries


def normalize_image(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())

    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def preprocess_image(image_u8: np.ndarray, mode: PreprocessMode) -> np.ndarray:
    if mode == "raw":
        return image_u8
    if mode == "equalize":
        return cv2.equalizeHist(image_u8)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image_u8)
    raise ValueError(f"Pre-processamento invalido: {mode}")


def load_dicom_histogram(
    path: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    hist_bins: int = DEFAULT_HIST_BINS,
    preprocess: PreprocessMode = "raw",
) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array

    if pixels.ndim > 2:
        pixels = pixels.squeeze()

    if pixels.ndim != 2:
        raise ValueError(f"Imagem invalida para {path}: dimensao {pixels.shape}")

    normalized = normalize_image(pixels)
    image_u8 = (normalized * 255.0).astype(np.uint8)
    resized = cv2.resize(
        image_u8,
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_AREA,
    )
    processed = preprocess_image(resized, preprocess)

    histogram = cv2.calcHist([processed], [0], None, [hist_bins], [0, 256])
    histogram = cv2.normalize(histogram, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return histogram.astype(np.float32)


def load_samples(
    base_dir: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    hist_bins: int = DEFAULT_HIST_BINS,
    preprocess: PreprocessMode = "raw",
) -> list[Sample]:
    pos_list = base_dir / "Pneumothorax_files.txt"
    neg_list = base_dir / "No_Pneumothorax_files.txt"

    raw_entries = parse_list_file(pos_list, POSITIVE_CLASS) + parse_list_file(
        neg_list, NEGATIVE_CLASS
    )

    samples: list[Sample] = []
    for rel_path, label in raw_entries:
        full_path = base_dir / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Imagem nao encontrada: {full_path}")
        histogram = load_dicom_histogram(
            full_path,
            image_size=image_size,
            hist_bins=hist_bins,
            preprocess=preprocess,
        )
        samples.append(Sample(path=full_path, label=label, histogram=histogram))

    return samples


def predict_label_by_hist_method(
    test_sample: Sample,
    candidates: list[Sample],
    *,
    method: HistMethod,
    strategy: Strategy = "nearest",
) -> str:
    if not candidates:
        raise ValueError("Lista de candidatos vazia para predicao")

    scores = [
        (
            float(cv2.compareHist(test_sample.histogram, candidate.histogram, method.flag)),
            candidate.label,
        )
        for candidate in candidates
    ]

    if strategy == "nearest":
        if method.higher_is_better:
            best_idx = int(np.argmax([score for score, _ in scores]))
        else:
            best_idx = int(np.argmin([score for score, _ in scores]))
        return scores[best_idx][1]

    positive_scores = [score for score, label in scores if label == POSITIVE_CLASS]
    negative_scores = [score for score, label in scores if label == NEGATIVE_CLASS]

    if strategy == "class_top3":
        if method.higher_is_better:
            positive_scores = sorted(positive_scores, reverse=True)[:3]
            negative_scores = sorted(negative_scores, reverse=True)[:3]
        else:
            positive_scores = sorted(positive_scores)[:3]
            negative_scores = sorted(negative_scores)[:3]

    if strategy not in ("class_mean", "class_top3"):
        raise ValueError(f"Estrategia invalida: {strategy}")

    positive_agg = float(np.mean(positive_scores))
    negative_agg = float(np.mean(negative_scores))

    if method.higher_is_better:
        return POSITIVE_CLASS if positive_agg >= negative_agg else NEGATIVE_CLASS
    return POSITIVE_CLASS if positive_agg <= negative_agg else NEGATIVE_CLASS


def evaluate_leave_one_out(
    samples: list[Sample],
    *,
    method: HistMethod,
    strategy: Strategy = "nearest",
) -> dict[str, float | int | str]:
    tp = tn = fp = fn = 0

    for idx, test_sample in enumerate(samples):
        candidates = samples[:idx] + samples[idx + 1 :]
        pred = predict_label_by_hist_method(
            test_sample,
            candidates,
            method=method,
            strategy=strategy,
        )
        true = test_sample.label

        if true == POSITIVE_CLASS and pred == POSITIVE_CLASS:
            tp += 1
        elif true == NEGATIVE_CLASS and pred == NEGATIVE_CLASS:
            tn += 1
        elif true == NEGATIVE_CLASS and pred == POSITIVE_CLASS:
            fp += 1
        elif true == POSITIVE_CLASS and pred == NEGATIVE_CLASS:
            fn += 1

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "Method": method.name,
        "Strategy": strategy,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def print_results(results: dict[str, float | int | str]) -> None:
    print(f"=== Metodo: {results['Method']} ===")
    print(f"Estrategia de decisao: {results['Strategy']}")
    print("=== Matriz de confusao (classe positiva: Pneumothorax) ===")
    print(f"TP: {results['TP']}")
    print(f"TN: {results['TN']}")
    print(f"FP: {results['FP']}")
    print(f"FN: {results['FN']}")
    print()
    print("=== Metricas ===")
    print(f"Sensibilidade: {results['Sensitivity']:.4f}")
    print(f"Especificidade: {results['Specificity']:.4f}")
    print()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    for method in HIST_METHODS:
        config = METHOD_CONFIGS.get(
            method.name,
            MethodConfig(
                image_size=DEFAULT_IMAGE_SIZE,
                hist_bins=DEFAULT_HIST_BINS,
                preprocess="raw",
                strategy="nearest",
            ),
        )

        samples = load_samples(
            base_dir,
            image_size=config.image_size,
            hist_bins=config.hist_bins,
            preprocess=config.preprocess,
        )
        if len(samples) != 30:
            print(f"Aviso: esperado 30 amostras, encontrado {len(samples)}")

        results = evaluate_leave_one_out(
            samples,
            method=method,
            strategy=config.strategy,
        )
        results["Config"] = (
            f"size={config.image_size[0]}x{config.image_size[1]}, "
            f"bins={config.hist_bins}, preprocess={config.preprocess}"
        )
        print_results(results)
        print(f"Configuracao usada: {results['Config']}")
        print()

if __name__ == "__main__":
    main()