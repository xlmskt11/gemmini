#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path


PACK_MAGIC = b"GMINIPK1"
PACK_VERSION = 4
PACK_ENDIAN = 0x01020304
PACK_FLAG_HYBRID = 1 << 0
PACK_LAYOUT_ROW_MAJOR = 0
PACK_LAYOUT_PAGE_B = 1
HYBRID_MAGIC = b"GMINIHY1"

FILE_HEADER = struct.Struct("<8sIIIIQ")
GEOMETRY_V4 = struct.Struct("<IIIIIIIIQ")
HYBRID_FOOTER = struct.Struct("<8sQ")


def parse_bool(value: str) -> bool:
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def read_exact(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size:
        raise RuntimeError(f"short artifact while reading {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--hybrid", required=True, type=parse_bool)
    parser.add_argument("--page-packed-b", required=True, type=parse_bool)
    args = parser.parse_args()

    if not args.model.is_file() or not args.pack.is_file():
        raise RuntimeError("model and weight-pack artifacts must both exist")

    header = FILE_HEADER.unpack(read_exact(args.pack, 0, FILE_HEADER.size))
    magic, version, endian, elem_size, scale_size, entry_count = header
    if magic != PACK_MAGIC or version != PACK_VERSION or endian != PACK_ENDIAN:
        raise RuntimeError("weight pack is not a little-endian v4 artifact")
    if elem_size != 2 or scale_size != 0 or entry_count == 0:
        raise RuntimeError("weight pack has invalid BF16 payload metadata")

    geometry = GEOMETRY_V4.unpack(
        read_exact(args.pack, FILE_HEADER.size, GEOMETRY_V4.size))
    (_, _, _, page_size, max_bytes, layout, flags, reserved,
     pack_fingerprint) = geometry
    expected_layout = PACK_LAYOUT_PAGE_B if args.page_packed_b else PACK_LAYOUT_ROW_MAJOR
    if page_size != 4096 or max_bytes == 0 or layout != expected_layout or reserved != 0:
        raise RuntimeError("weight-pack geometry does not match the selected artifact format")
    if bool(flags & PACK_FLAG_HYBRID) != args.hybrid or flags & ~PACK_FLAG_HYBRID:
        raise RuntimeError("weight-pack hybrid flags do not match the selected artifact format")

    model_size = args.model.stat().st_size
    if model_size < HYBRID_FOOTER.size:
        raise RuntimeError("model artifact is too short")
    footer_magic, model_fingerprint = HYBRID_FOOTER.unpack(
        read_exact(args.model, model_size - HYBRID_FOOTER.size, HYBRID_FOOTER.size))
    has_footer = footer_magic == HYBRID_MAGIC
    if args.hybrid:
        if not has_footer or model_fingerprint != pack_fingerprint:
            raise RuntimeError("thin GGUF footer and v4 pack fingerprint do not match")
    elif has_footer:
        raise RuntimeError("full-GGUF mode received a hybrid sparse model")

    print(
        "GEMMINI-HYBRID-ARTIFACTS-VERIFIED,"
        f"hybrid={int(args.hybrid)},page_packed_b={int(args.page_packed_b)},"
        f"entries={entry_count},fingerprint=0x{pack_fingerprint:016x}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, struct.error) as error:
        raise SystemExit(f"verify-hybrid-artifacts: {error}")
