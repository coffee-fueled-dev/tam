"""Canonical hash-chained ledger and seal utilities for Stage 1E."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_payload(payload: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(payload))


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return digest_payload(protocol)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def genesis_hash(protocol_digest: str, scenario: str, seed: int) -> str:
    return digest_payload({
        "kind": "genesis",
        "protocol_hash": protocol_digest,
        "scenario": scenario,
        "seed": seed,
    })


class LedgerWriter:
    def __init__(
        self,
        path: Path,
        protocol_digest: str,
        scenario: str,
        seed: int,
    ) -> None:
        self.path = path
        self.protocol_digest = protocol_digest
        self.scenario = scenario
        self.seed = seed
        self.previous_hash = genesis_hash(protocol_digest, scenario, seed)
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> str:
        payload = {
            **record,
            "previous_hash": self.previous_hash,
            "protocol_hash": self.protocol_digest,
            "scenario": self.scenario,
            "seed": self.seed,
            "sequence": self.count,
        }
        digest = digest_payload(payload)
        line = canonical_json({**payload, "record_hash": digest})
        self._handle.write(line + "\n")
        self._handle.flush()
        self.previous_hash = digest
        self.count += 1
        return digest

    def close(self) -> str:
        self._handle.close()
        return self.previous_hash


def read_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_ledger(
    path: Path,
    protocol_digest: str,
    scenario: str,
    seed: int,
) -> tuple[bool, str]:
    expected_previous = genesis_hash(protocol_digest, scenario, seed)
    try:
        rows = read_ledger(path)
    except OSError as exc:
        return False, f"unreadable ledger: {exc}"
    if not rows:
        return False, "empty ledger"
    if len(rows) % 2 != 0:
        return False, "truncated commit/resolution pair"
    for index, row in enumerate(rows):
        if row.get("sequence") != index:
            return False, f"sequence mismatch at {index}"
        if row.get("scenario") != scenario or row.get("seed") != seed:
            return False, f"identity mismatch at {index}"
        if row.get("protocol_hash") != protocol_digest:
            return False, f"protocol hash mismatch at {index}"
        if row.get("previous_hash") != expected_previous:
            return False, f"chain break at {index}"
        payload = {key: value for key, value in row.items() if key != "record_hash"}
        digest = digest_payload(payload)
        if digest != row.get("record_hash"):
            return False, f"record hash mismatch at {index}"
        expected_previous = digest
    return True, expected_previous


def build_seal(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    code_hashes: Mapping[str, str],
    run_final_hashes: Mapping[str, str],
    event_counts: Mapping[str, int],
    final_state_hashes: Mapping[str, str],
    summary_hash: str,
) -> dict[str, Any]:
    seal = {
        "kind": "stage1e_seal",
        "protocol_hash": protocol_digest,
        "protocol_name": protocol.get("name"),
        "code_hashes": dict(sorted(code_hashes.items())),
        "run_final_hashes": dict(sorted(run_final_hashes.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "final_state_hashes": dict(sorted(final_state_hashes.items())),
        "summary_hash": summary_hash,
    }
    seal["seal_hash"] = digest_payload(seal)
    return seal


def write_seal(path: Path, seal: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path = path.with_suffix(".txt")
    text_path.write_text(
        "\n".join([
            f"seal_hash={seal['seal_hash']}",
            f"protocol_hash={seal['protocol_hash']}",
            f"summary_hash={seal['summary_hash']}",
            "Retain this file outside artifacts/ for tamper evidence.",
            "",
        ]),
        encoding="utf-8",
    )


def code_file_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in names}
