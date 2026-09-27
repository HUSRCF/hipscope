#!/usr/bin/env python3
"""Export RDNA4 XML instruction encodings for the offline pm-census binary.

Usage: python3 crates/peacemaker-lift/tools/pm-census-xml.py amdgpu_isa_rdna4.xml /scratch/rdna4-encodings.tsv
Uses only Python's standard-library XML parser; the input SHA is pinned by pm-census.
"""
import hashlib
import pathlib
import sys
import xml.etree.ElementTree as ET

PIN = "f8a290c8471e26a1071b08b61a33e4d9efa46ec6cedcdb8e5ba57d7969b60692"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: pm-census-xml.py <amdgpu_isa_rdna4.xml> <output.tsv>")
    source, output = map(pathlib.Path, sys.argv[1:])
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PIN:
        raise SystemExit(f"RDNA4 XML SHA-256 {digest} does not equal pinned {PIN}")
    root = ET.fromstring(data)
    encodings = {}
    for encoding in root.findall("ISA/Encodings/Encoding"):
        name = encoding.findtext("EncodingName")
        encodings[name] = (
            int(encoding.findtext("BitCount")),
            int(encoding.findtext("EncodingIdentifierMask"), 2),
            sorted({int(node.text, 2) for node in encoding.findall("EncodingIdentifiers/EncodingIdentifier")}),
        )
    rows = [f"# xml_sha256={digest}", "encoding\topcode\tbits\tspelling\tid_mask\tids"]
    for instruction in root.findall("ISA/Instructions/Instruction"):
        name = instruction.findtext("InstructionName").lower()
        for variant in instruction.findall("InstructionEncodings/InstructionEncoding"):
            encoding = variant.findtext("EncodingName")
            bits, mask, identifiers = encodings[encoding]
            fields = (encoding, variant.findtext("Opcode"), str(bits), name,
                      f"{mask:x}", ",".join(f"{identifier:x}" for identifier in identifiers))
            rows.append("\t".join(fields))
    output.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)-2} XML instruction-encoding entries -> {output}")


if __name__ == "__main__":
    main()
