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
    if v in {"had", "hadamard", "kron", "random"}:
        return v
    raise ValueError(f"unsupported rotation={value!r}; expected none|hadamard|random|had|kron.")


def normalize_solver(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"gptq", "ldlq", "vamp", "ldlq_tune10"}:
        return v
    raise ValueError(f"unsupported solver={value!r}; expected gptq|ldlq|vamp|ldlq_tune10.")


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
    combo = QuantCombo(
        backend=normalize_backend(backend),
        codebook=normalize_codebook(codebook),
        preprocess=normalize_preprocess(preprocess),
        rotation=normalize_rotation(rotation),
        solver=normalize_solver(solver),
        polish_mode=normalize_polish_mode(polish_mode),
        topk_source=str(topk_source or "aux_vamp").strip().lower(),
    )
    validate_combo(combo)
    return combo


def validate_combo(combo: QuantCombo) -> None:
    if combo.backend == "gptqmodel":
        if combo.codebook != "grid_g128":
            raise ValueError("gptqmodel backend only supports codebook=grid_g128.")
        if combo.solver not in {"gptq", "vamp"}:
            raise ValueError("gptqmodel backend only supports solver in {gptq, vamp}.")
        if combo.preprocess not in {"none", "awq"}:
            raise ValueError("gptqmodel backend only supports preprocess in {none, awq}.")
        if combo.rotation not in {"none", "hadamard", "random"}:
            raise ValueError("gptqmodel backend only supports rotation in {none, hadamard, random}.")
    elif combo.backend == "quip":
        if combo.codebook != "e8p12":
            raise ValueError("quip backend only supports codebook=e8p12.")
        if combo.preprocess != "none":
            raise ValueError("quip backend does not support preprocess != none.")
        if combo.solver not in {"ldlq", "vamp", "ldlq_tune10"}:
            raise ValueError("quip backend only supports solver in {ldlq, vamp, ldlq_tune10}.")
        if combo.rotation not in {"had", "kron"}:
            raise ValueError("quip backend only supports rotation in {had, kron}.")

    if combo.solver == "ldlq_tune10" and combo.polish_mode != "none":
        raise ValueError("solver=ldlq_tune10 requires polish_mode=none.")
    if combo.polish_mode != "topk_block_cd" and combo.topk_source != "aux_vamp":
        raise ValueError("topk_source is only meaningful for polish_mode=topk_block_cd.")

