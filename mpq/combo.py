from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantCombo:
    backend: str
    codebook: str
    preprocess: str
    rotation: str
    solver: str
    polish_mode: str
    topk_source: str = "aux_vamp"

    def label(self) -> str:
        return "__".join(
            [
                f"backend-{self.backend}",
                f"codebook-{self.codebook}",
                f"pre-{self.preprocess}",
                f"rot-{self.rotation}",
                f"solver-{self.solver}",
                f"polish-{self.polish_mode}",
            ]
        )


def normalize_backend(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"gptq", "gptqmodel"}:
        return "gptqmodel"
    if v in {"quip", "quip-sharp", "quip_sharp"}:
        return "quip"
    raise ValueError(f"unsupported backend={value!r}; expected gptqmodel|quip.")


def normalize_codebook(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"grid", "grid_g128", "naive_grid", "uniform"}:
        return "grid_g128"
    if v in {"e8p12", "quip_e8p12"}:
        return "e8p12"
    raise ValueError(f"unsupported codebook={value!r}; expected grid_g128|e8p12.")


def normalize_preprocess(value: str) -> str:
    v = str(value or "none").strip().lower()
    if v in {"", "none", "off"}:
        return "none"
    if v == "awq":
        return "awq"
    raise ValueError(f"unsupported preprocess={value!r}; expected none|awq.")


def normalize_rotation(value: str) -> str:
    v = str(value or "none").strip().lower()
    if v in {"", "none", "off"}:
        return "none"
    if v in {"had", "kron"}:
        return v
    if v == "model_hadamard":
        return "model_hadamard"
    if v == "model_random":
        return "model_random"
    if v == "internal_hadamard":
        return "internal_hadamard"
    raise ValueError(
        "unsupported rotation="
        f"{value!r}; expected none|model_hadamard|model_random|internal_hadamard|had|kron."
    )


def normalize_solver(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"gptq", "ldlq", "vamp", "ldlq_tune10", "cdquant", "quantease", "ojbkq"}:
        return v
    raise ValueError(f"unsupported solver={value!r}; expected gptq|ldlq|vamp|ldlq_tune10|cdquant|quantease|ojbkq.")


def normalize_polish_mode(value: str) -> str:
    v = str(value or "none").strip().lower()
    if v in {"", "none", "off"}:
        return "none"
    if v in {"block_cd", "cd", "full_cd"}:
        return "block_cd"
    if v in {"topk_block_cd", "topk_cd", "topk"}:
        return "topk_block_cd"
    raise ValueError(f"unsupported polish_mode={value!r}; expected none|block_cd|topk_block_cd.")


def make_combo(
    *,
    backend: str,
    codebook: str,
    preprocess: str,
    rotation: str,
    solver: str,
    polish_mode: str,
    topk_source: str = "aux_vamp",
) -> QuantCombo:
    backend_norm = normalize_backend(backend)
    codebook_norm = normalize_codebook(codebook)
    preprocess_norm = normalize_preprocess(preprocess)
    solver_norm = normalize_solver(solver)
    rotation_norm = normalize_rotation(rotation)
    combo = QuantCombo(
        backend=backend_norm,
        codebook=codebook_norm,
        preprocess=preprocess_norm,
        rotation=rotation_norm,
        solver=solver_norm,
        polish_mode=normalize_polish_mode(polish_mode),
        topk_source=str(topk_source or "aux_vamp").strip().lower(),
    )
    validate_combo(combo)
    return combo


def validate_combo(combo: QuantCombo) -> None:
    if combo.backend == "gptqmodel":
        if combo.codebook != "grid_g128":
            raise ValueError("gptqmodel backend only supports codebook=grid_g128.")
        if combo.solver not in {"gptq", "vamp", "cdquant", "quantease", "ojbkq"}:
            raise ValueError("gptqmodel backend only supports solver in {gptq, vamp, cdquant, quantease, ojbkq}.")
        if combo.preprocess not in {"none", "awq"}:
            raise ValueError("gptqmodel backend only supports preprocess in {none, awq}.")
        if combo.rotation not in {"none", "model_hadamard", "model_random", "internal_hadamard"}:
            raise ValueError(
                "gptqmodel backend only supports rotation in "
                "{none, model_hadamard, model_random, internal_hadamard}."
            )
    elif combo.backend == "quip":
        if combo.codebook != "e8p12":
            raise ValueError("quip backend only supports codebook=e8p12.")
        if combo.preprocess not in {"none", "awq"}:
            raise ValueError("quip backend only supports preprocess in {none, awq}.")
        if combo.solver not in {"ldlq", "vamp", "ldlq_tune10", "cdquant", "quantease", "ojbkq"}:
            raise ValueError("quip backend only supports solver in {ldlq, vamp, ldlq_tune10, cdquant, quantease, ojbkq}.")
        if combo.rotation not in {"had", "kron"}:
            raise ValueError("quip backend only supports rotation in {had, kron}.")

    if combo.solver == "ldlq_tune10" and combo.polish_mode != "none":
        raise ValueError("solver=ldlq_tune10 requires polish_mode=none.")
    if combo.solver == "ojbkq" and combo.polish_mode not in {"none", "block_cd"}:
        raise ValueError("solver=ojbkq supports polish_mode in {none, block_cd}.")
    if combo.polish_mode != "topk_block_cd" and combo.topk_source != "aux_vamp":
        raise ValueError("topk_source is only meaningful for polish_mode=topk_block_cd.")
