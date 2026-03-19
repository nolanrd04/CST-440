from __future__ import annotations

from pathlib import Path


def tflite_to_c_header(tflite_path: Path, header_path: Path, var_name: str = "g_model") -> Path:
    """Step 9a: Convert .tflite binary into a C header byte array."""
    data = tflite_path.read_bytes()
    header_path.parent.mkdir(parents=True, exist_ok=True)

    hex_bytes = ", ".join(f"0x{b:02x}" for b in data)
    content = (
        "#pragma once\n\n"
        f"const unsigned char {var_name}[] = {{ {hex_bytes} }};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )
    header_path.write_text(content, encoding="utf-8")
    return header_path


def estimate_tensor_arena_bytes(model_size_bytes: int) -> int:
    """Step 8e/9b heuristic for initial tensor arena allocation."""
    return int(model_size_bytes * 2.5)
