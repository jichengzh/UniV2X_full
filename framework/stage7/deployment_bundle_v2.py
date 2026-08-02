"""Immutable, exact-file deployment bundle admission for Stage7 v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence

from framework.stage7.core_ablation_v2 import canonical_sha256


DEPLOYMENT_DIR = "deployment"
CODE_DIR = "code"
MANIFEST_NAME = "deployment_manifest_v2.json"
STATE_NAME = "deployment_state_v2.json"
SCHEMA = "stage7_core_ablation_deployment_bundle_v2"
STATE_SCHEMA = "stage7_core_ablation_deployment_bundle_state_v2"
OVERLAY_FILES = frozenset({
    "framework/__init__.py",
    "framework/stage5/__init__.py",
})
EXECUTABLE_BUNDLE_FILES = frozenset({
    "scripts/stage5_materialize_round_sources_v1.sh",
})
DEFAULT_BUNDLE_FILES = (
    "framework/__init__.py",
    "framework/stage5/__init__.py",
    "framework/stage5/genome_contract_v1.py",
    "framework/stage5/measurement_plan_v1.py",
    "framework/stage5/measurement_plan_v2.py",
    "framework/stage7/__init__.py",
    "framework/stage7/ablation_statistics_v2.py",
    "framework/stage7/actual_mode_v2.py",
    "framework/stage7/actual_pipeline_support_v2.py",
    "framework/stage7/actual_pipeline_v2.py",
    "framework/stage7/actual_v3_adapter_v2.py",
    "framework/stage7/actual_v3_selector_v2.py",
    "framework/stage7/core_ablation_v2.py",
    "framework/stage7/core_cache_v2.py",
    "framework/stage7/deployment_bundle_v2.py",
    "framework/stage7/executor_admission_v2.py",
    "framework/stage7/formal_evidence_v2.py",
    "framework/stage7/formal_inputs_v2.py",
    "framework/stage7/no_gpu_support_v2.py",
    "framework/stage7/online_component_ablation_v1.py",
    "framework/stage7/paper_outputs_v2.py",
    "framework/stage7/physical_execution_v2.py",
    "framework/stage7/physical_feedback_v2.py",
    "framework/stage7/physical_runtime_v2.py",
    "framework/stage7/prepare_state_v2.py",
    "framework/stage7/search_policy_v1.py",
    "framework/stage7/source_execution_v2.py",
    "framework/stage7/source_relocation_v2.py",
    "framework/stage7/source_resolution_v2.py",
    "framework/stage7/source_round_orchestration_v2.py",
    "scripts/stage5_materialize_round_sources_v1.sh",
    "scripts/stage7_actual_feedback_barrier_v2.py",
    "scripts/stage7_ablation_scheduler_v1.py",
    "scripts/stage7_core_ablation_orchestrator_v2.py",
    "scripts/stage7_core_ablation_scheduler_v2.py",
    "scripts/stage7_core_round_controller_v2.sh",
    "scripts/stage7_core_round_worker_v2.py",
    "scripts/stage7_core_online_ablation_v2.py",
    "scripts/stage7_finalize_ablation_v1.py",
    "scripts/stage7_finalize_core_ablation_v2.py",
    "scripts/stage7_h800_runtime_v2.py",
    "scripts/stage7_orchestrator_scheduler_adapter_v2.py",
    "scripts/stage7_prepare_core_ablation_v2.py",
    "scripts/stage7_resolve_round_sources_v2.py",
    "scripts/stage7_scheduler_requests_v2.py",
    "scripts/stage7_source_lease_controller_v2.py",
    "scripts/stage7_source_scheduler_v2.py",
    "scripts/stage7_execute_actual_v3_misses_v2.sh",
    "scripts/stage7_core_no_gpu_dry_run_v2.py",
)
DEFAULT_PRIMITIVE_FILES = (
    "scripts/stage5_task_round_controller_v3.sh",
    "scripts/stage5_build_performance_plan_v2.py",
    "scripts/stage35_gold32_performance_plan_v1.py",
    "framework/stage5/independent_validation_v1.py",
    "scripts/stage3_tvm_int8_quant_contract_v3.py",
    "scripts/stage3_execute_performance_plan_v3.py",
    "scripts/stage5_ap_plan_v2.py",
    "scripts/stage3_execute_ap_plan_v3.py",
    "scripts/stage5_finalize_feedback_v2.py",
    "scripts/stage3_finalize_gold96_v3.py",
    "scripts/stage5_promote_actual_feedback_v3.py",
)
EXPECTED_PRIMITIVE_SHA256 = {
    "scripts/stage5_task_round_controller_v3.sh": "132321077fc1a308dc43fe3e5af3f6d74725088fb526e05e57bd31530ed40170",
    "scripts/stage5_build_performance_plan_v2.py": "99126d0a3637f9cb51d8210b265c4d3133973a68bb3f11858966889771950912",
    "scripts/stage35_gold32_performance_plan_v1.py": "f42b2c7df7d205ed07f67872207d9dddd1336a8e59dfa64b4bf6db2fbf1412a0",
    "framework/stage5/independent_validation_v1.py": "f7460d9ada6cc2f9c540d2c3f7612f8071f65cfc1242e3b21647ee548296a4c3",
    "scripts/stage3_tvm_int8_quant_contract_v3.py": "788556b775a1378e75458623c9c3186a417d30609200979694079ec628536fe5",
    "scripts/stage3_execute_performance_plan_v3.py": "3c0af3d857570a2bdba6b6ff5fe50a532abb5e079370119923ff0dae16f57335",
    "scripts/stage5_ap_plan_v2.py": "eaf175f0ce918a72924ea067d55ef1a2f8e8df3f0b54fa4afd984df9ba44c636",
    "scripts/stage3_execute_ap_plan_v3.py": "da71d1ba94480b9034c9dc3b16011ba0bba83166e478f865934b80b50dbecaff",
    "scripts/stage5_finalize_feedback_v2.py": "9c22be5270b70b5dd7cf3e3e8eede3e3ea8aecfe89b5c8d86893e42340b37393",
    "scripts/stage3_finalize_gold96_v3.py": "21ea511c2f27874c612b85b41e028c63123490433f1d88dd4498f653e04cc238",
    "scripts/stage5_promote_actual_feedback_v3.py": "fbbd295cc881ebac6db50ee3764eb667bc921168a1981f96ec7c9d3fcb440187",
}
EXPECTED_CROSS_HOST_SOURCE_SHA256 = {
    "framework/stage5/__init__.py":
        "7d1ae81843e76eefa32d30e01f9f811b0eb2481f77cfaac19b502a0d3755e7d5",
    "framework/stage5/genome_contract_v1.py":
        "1aa183c576510cf4612d3a96028a56e37b"
        "8fd4281058867b7d3b4a08660fbe7b",
    "framework/stage5/measurement_plan_v1.py":
        "7da96d705831ec834cee6d9e519753ba0c7c587b78e13a81ae6761f8dc67b15c",
    "framework/stage5/measurement_plan_v2.py":
        "4395418d9a0aedf4668bf045e1e23883779971ad823d75ef7a7ec92cc41196b8",
}
EXPECTED_CROSS_HOST_BUNDLE_SHA256 = {
    "framework/stage5/__init__.py":
        "9ab1e041f6521c9f7a632a3fff30049a71b0395917f08d13be7cead1d7a94419",
    "framework/stage5/genome_contract_v1.py":
        "1aa183c576510cf4612d3a96028a56e37b"
        "8fd4281058867b7d3b4a08660fbe7b",
    "framework/stage5/measurement_plan_v1.py":
        "7da96d705831ec834cee6d9e519753ba0c7c587b78e13a81ae6761f8dc67b15c",
    "framework/stage5/measurement_plan_v2.py":
        "4395418d9a0aedf4668bf045e1e23883779971ad823d75ef7a7ec92cc41196b8",
}
_CROSS_HOST_BOOTSTRAP_PATHS = frozenset({
    "framework/__init__.py",
    "framework/stage5/__init__.py",
})
_SECRET_PATTERN = re.compile(
    r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}",
    re.IGNORECASE,
)


def primitive_pins_sha256() -> str:
    """Hash the exact ordered primitive contract independently of host paths."""
    return canonical_sha256(
        [
            {
                "repo_relative_path": relative,
                "sha256": EXPECTED_PRIMITIVE_SHA256[relative],
            }
            for relative in DEFAULT_PRIMITIVE_FILES
        ]
    )


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_external_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"external deployment pins are missing or invalid: {label}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"deployment {label} is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"deployment {label} escapes its root")
    return path


def _under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"deployment {label} escapes its root") from error
    return resolved


def _reject_link_or_unsafe_mode(path: Path, *, label: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"deployment {label} must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise ValueError(f"deployment {label} has hardlink drift")
    if metadata.st_mode & stat.S_IWOTH:
        raise ValueError(f"deployment {label} is world-writable")
    return metadata


def _scan_for_secrets(content: bytes, *, label: str) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"deployment code is not UTF-8: {label}") from error
    if "-----BEGIN " in text and ("PRIVATE" + " KEY-----") in text:
        raise ValueError(f"deployment contains private-key material: {label}")
    if (
        _SECRET_PATTERN.search(text)
        or re.search(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{12,}", text)
        or re.search(
            r"AKIA[0-9A-Z]{16}|AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY)\s*[:=]", text
        )
    ):
        raise ValueError(f"deployment contains credential-like material: {label}")


def _overlay_bytes(package: str, source: Path) -> bytes:
    if package == "framework/__init__.py":
        prefix = b'"""Stage7 deployment overlay package."""\n'
    else:
        prefix = source.read_bytes()
    return (
        prefix
        + b"from pkgutil import extend_path\n\n"
        + b"__path__ = extend_path(__path__, __name__)\n"
    )


def _bundle_content(source: Path, relative: str) -> bytes:
    return (
        _overlay_bytes(relative, source)
        if relative in OVERLAY_FILES
        else source.read_bytes()
    )


def _bundle_mode(source_mode: int, relative: str) -> int:
    mode = stat.S_IMODE(source_mode) & ~stat.S_IWOTH
    if relative in EXECUTABLE_BUNDLE_FILES:
        mode |= (mode & 0o444) >> 2
    return mode


def _atomic_write(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        metadata = _reject_link_or_unsafe_mode(path, label="existing artifact")
        if metadata.st_uid != os.getuid():
            raise ValueError("deployment existing artifact owner drift")
        if path.read_bytes() == content:
            return
        raise ValueError(f"refusing to overwrite deployment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        metadata = _reject_link_or_unsafe_mode(
            temporary, label="temporary artifact"
        )
        if metadata.st_uid != os.getuid() or temporary.read_bytes() != content:
            raise ValueError("deployment temporary artifact drift")
        os.replace(temporary, path)
        descriptor = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_owned(path: Path, content: bytes) -> None:
    metadata = _reject_link_or_unsafe_mode(path, label="rebind artifact")
    if metadata.st_uid != os.getuid():
        raise ValueError("deployment rebind artifact owner drift")
    temporary = path.with_name(path.name + ".rebind.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("deployment rebind temporary artifact already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _identity(owner: str | None, owner_uid: int | None) -> tuple[str, int]:
    uid = os.getuid() if owner_uid is None else owner_uid
    if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
        raise ValueError("deployment owner UID is invalid")
    username = pwd.getpwuid(uid).pw_name if owner is None else owner
    if not isinstance(username, str) or not username:
        raise ValueError("deployment owner is invalid")
    return username, uid


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"deployment {label} is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"deployment {label} must be an object")
    return dict(payload)


def _required_dirs(root: Path, files: Iterable[Path]) -> set[Path]:
    result = {root}
    for path in files:
        parent = path.parent
        while parent != root:
            result.add(parent)
            parent = parent.parent
    return result


def _manifest_unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(manifest)
    unsigned.pop("manifest_unsigned_payload_sha256", None)
    unsigned.pop("deployment_bundle_sha256", None)
    return unsigned


def _release_records(files: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "repo_relative_path": str(record["repo_relative_path"]),
            "sha256": str(record["sha256"]),
        }
        for record in files
    ]


def _cross_host_override_records() -> list[dict[str, str]]:
    if set(EXPECTED_CROSS_HOST_SOURCE_SHA256) != set(
        EXPECTED_CROSS_HOST_BUNDLE_SHA256
    ):
        raise ValueError("deployment cross-host override contract is incomplete")
    return [
        {
            "repo_relative_path": relative,
            "reviewed_source_sha256":
                EXPECTED_CROSS_HOST_SOURCE_SHA256[relative],
            "bundle_sha256": EXPECTED_CROSS_HOST_BUNDLE_SHA256[relative],
        }
        for relative in EXPECTED_CROSS_HOST_SOURCE_SHA256
    ]


def _cross_host_source_mismatch_allowed(relative: str) -> bool:
    if relative not in DEFAULT_BUNDLE_FILES:
        return False
    return (
        relative in _CROSS_HOST_BOOTSTRAP_PATHS
        or relative in EXPECTED_CROSS_HOST_SOURCE_SHA256
        or relative.startswith("framework/stage7/")
        or relative.startswith("scripts/stage7_")
    )


def build_deployment_bundle(
    *,
    v2_root: Path,
    frozen_repo_root: Path,
    bundle_files: Sequence[str] = DEFAULT_BUNDLE_FILES,
    primitive_files: Sequence[str] = DEFAULT_PRIMITIVE_FILES,
    owner: str | None = None,
    owner_uid: int | None = None,
) -> dict[str, Any]:
    """Copy a deliberately small code overlay into a deployment-only v2 root."""
    if tuple(bundle_files) != DEFAULT_BUNDLE_FILES:
        raise ValueError("deployment requires the exact reviewed deployment allowlist")
    if tuple(primitive_files) != DEFAULT_PRIMITIVE_FILES:
        raise ValueError("deployment requires the complete ordered primitive set")
    supplied_root = Path(v2_root)
    if not supplied_root.is_absolute() or supplied_root != supplied_root.resolve(
        strict=False
    ):
        raise ValueError("deployment v2_root must be canonical and non-symlinked")
    root = supplied_root
    frozen = Path(frozen_repo_root).resolve(strict=True)
    if not root.is_absolute() or not frozen.is_absolute() or not frozen.is_dir():
        raise ValueError("deployment roots must be absolute directories")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("deployment bundle root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    username, uid = _identity(owner, owner_uid)
    code_root = root / DEPLOYMENT_DIR / CODE_DIR
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative_text in bundle_files:
        relative = _safe_relative_path(relative_text, label="source path")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError("deployment bundle has duplicate destination")
        seen.add(normalized)
        source = _under(frozen / relative, frozen, label="source path")
        source_metadata = _reject_link_or_unsafe_mode(source, label="source")
        reviewed_source_sha = EXPECTED_CROSS_HOST_SOURCE_SHA256.get(normalized)
        if (
            reviewed_source_sha is not None
            and _file_sha256(source) != reviewed_source_sha
        ):
            raise ValueError(
                "deployment cross-host override source SHA is not reviewed"
            )
        content = _bundle_content(source, normalized)
        reviewed_bundle_sha = EXPECTED_CROSS_HOST_BUNDLE_SHA256.get(normalized)
        if (
            reviewed_bundle_sha is not None
            and _sha256_bytes(content) != reviewed_bundle_sha
        ):
            raise ValueError(
                "deployment cross-host override bundle SHA is not reviewed"
            )
        _scan_for_secrets(content, label=normalized)
        destination = _under(code_root / relative, code_root, label="destination")
        _atomic_write(destination, content)
        os.chmod(destination, _bundle_mode(source_metadata.st_mode, normalized))
        records.append(
            {
                "repo_relative_path": normalized,
                "source_path": str(source),
                "destination_path": str(destination),
                "mode": stat.S_IMODE(destination.stat().st_mode),
                "byte_count": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    files_sha = canonical_sha256(records)
    release_sha256 = canonical_sha256(_release_records(records))
    primitive_records: list[dict[str, str]] = []
    for relative_text in primitive_files:
        relative = _safe_relative_path(relative_text, label="primitive path")
        source = _under(frozen / relative, frozen, label="primitive path")
        _reject_link_or_unsafe_mode(source, label="primitive")
        actual = _file_sha256(source)
        if (
            relative.as_posix() not in EXPECTED_PRIMITIVE_SHA256
            or actual != EXPECTED_PRIMITIVE_SHA256[relative.as_posix()]
        ):
            raise ValueError("deployment primitive SHA is not formally pinned")
        primitive_records.append(
            {
                "repo_relative_path": relative.as_posix(),
                "path": str(source),
                "sha256": actual,
            }
        )
    unsigned = {
        "schema_version": SCHEMA,
        "v2_root": str(root),
        "bundle_code_root": str(code_root),
        "frozen_repo_root": str(frozen),
        "owner": username,
        "owner_uid": uid,
        "creator_pid": os.getpid(),
        "files": records,
        "frozen_primitives": primitive_records,
        "cross_host_source_overrides": _cross_host_override_records(),
        "ordered_file_list_sha256": files_sha,
        "deployment_release_sha256": release_sha256,
        "contains_secrets": False,
        "secret_scan": {"credential_like_matches": 0, "private_key_matches": 0},
        "remote_repo_write_count": 0,
        "tmp_write_count": 0,
    }
    manifest = {
        **unsigned,
        "manifest_unsigned_payload_sha256": canonical_sha256(unsigned),
        "deployment_bundle_sha256": canonical_sha256(unsigned),
    }
    manifest_path = root / DEPLOYMENT_DIR / MANIFEST_NAME
    _atomic_write(manifest_path, _json_bytes(manifest))
    manifest_file_sha256 = _file_sha256(manifest_path)
    state_unsigned = {
        "schema_version": STATE_SCHEMA,
        "status": "canonical_pre_prepare_bundle",
        "owner": username,
        "owner_uid": uid,
        "deployment_manifest_sha256": manifest["manifest_unsigned_payload_sha256"],
        "deployment_manifest_file_sha256": manifest_file_sha256,
        "deployment_bundle_sha256": manifest["deployment_bundle_sha256"],
        "deployment_release_sha256": release_sha256,
        "actual_directory_manifest_sha256": canonical_sha256(
            sorted(
                [MANIFEST_NAME, STATE_NAME]
                + [record["repo_relative_path"] for record in records]
            )
        ),
    }
    _atomic_write(
        root / DEPLOYMENT_DIR / STATE_NAME,
        _json_bytes(
            {**state_unsigned, "state_sha256": canonical_sha256(state_unsigned)}
        ),
    )
    return validate_deployment_bundle(
        root,
        frozen_repo_root=frozen,
        expected_release_sha256=release_sha256,
        expected_manifest_file_sha256=manifest_file_sha256,
    )


def _validate_deployment_bundle(
    v2_root: Path,
    *,
    frozen_repo_root: Path,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_owner: str | None = None,
    expected_owner_uid: int | None = None,
    filesystem_owner_uid: int | None = None,
) -> dict[str, Any]:
    """Authenticate the exact pre-prepare deployment tree without trusting paths."""
    release_pin = _require_external_sha256(
        expected_release_sha256, label="expected release SHA256"
    )
    manifest_file_pin = _require_external_sha256(
        expected_manifest_file_sha256, label="expected manifest-file SHA256"
    )
    root = Path(v2_root).resolve(strict=True)
    frozen = Path(frozen_repo_root).resolve(strict=True)
    deployment = root / DEPLOYMENT_DIR
    manifest_path = deployment / MANIFEST_NAME
    state_path = deployment / STATE_NAME
    if (
        not deployment.is_dir()
        or not manifest_path.is_file()
        or not state_path.is_file()
    ):
        raise ValueError("deployment bundle manifest/state is missing")
    if _file_sha256(manifest_path) != manifest_file_pin:
        raise ValueError("deployment external manifest-file pin mismatch")
    manifest = _read_mapping(manifest_path, label="manifest")
    state_payload = _read_mapping(state_path, label="state")
    manifest_metadata = _reject_link_or_unsafe_mode(manifest_path, label="manifest")
    state_metadata = _reject_link_or_unsafe_mode(state_path, label="state")
    unsigned = _manifest_unsigned(manifest)
    if manifest.get("deployment_release_sha256") != release_pin:
        raise ValueError("deployment external release pin mismatch")
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("manifest_unsigned_payload_sha256")
        != canonical_sha256(unsigned)
        or manifest.get("deployment_bundle_sha256") != canonical_sha256(unsigned)
        or manifest.get("v2_root") != str(root)
        or manifest.get("bundle_code_root") != str(deployment / CODE_DIR)
        or manifest.get("frozen_repo_root") != str(frozen)
        or manifest.get("contains_secrets") is not False
        or manifest.get("remote_repo_write_count") != 0
        or manifest.get("tmp_write_count") != 0
    ):
        raise ValueError("deployment manifest identity drift")
    owner = manifest.get("owner")
    uid = manifest.get("owner_uid")
    if not isinstance(owner, str) or not isinstance(uid, int) or isinstance(uid, bool):
        raise ValueError("deployment manifest owner drift")
    artifact_uid = uid if filesystem_owner_uid is None else filesystem_owner_uid
    if (
        isinstance(artifact_uid, bool)
        or not isinstance(artifact_uid, int)
        or artifact_uid < 0
    ):
        raise ValueError("deployment filesystem owner UID is invalid")
    if (
        manifest_metadata.st_uid != artifact_uid
        or state_metadata.st_uid != artifact_uid
    ):
        raise ValueError("deployment metadata owner drift")
    if (expected_owner is not None and owner != expected_owner) or (
        expected_owner_uid is not None and uid != expected_owner_uid
    ):
        raise ValueError("deployment manifest owner drift")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or not files
        or manifest.get("ordered_file_list_sha256") != canonical_sha256(files)
    ):
        raise ValueError("deployment manifest file list drift")
    if [
        record.get("repo_relative_path") if isinstance(record, Mapping) else None
        for record in files
    ] != list(DEFAULT_BUNDLE_FILES):
        raise ValueError("deployment manifest exact allowlist drift")
    if manifest.get("cross_host_source_overrides") != (
        _cross_host_override_records()
    ):
        raise ValueError("deployment cross-host override contract drift")
    code_root = deployment / CODE_DIR
    primitives = manifest.get("frozen_primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError("deployment manifest primitive pins are missing")
    if [
        record.get("repo_relative_path") if isinstance(record, Mapping) else None
        for record in primitives
    ] != list(DEFAULT_PRIMITIVE_FILES):
        raise ValueError("deployment primitive set/order drift")
    primitive_paths: set[Path] = set()
    for record in primitives:
        if not isinstance(record, Mapping) or set(record) != {
            "repo_relative_path",
            "path",
            "sha256",
        }:
            raise ValueError("deployment primitive pin drift")
        relative = _safe_relative_path(record["repo_relative_path"], label="primitive")
        path = _under(frozen / relative, frozen, label="primitive")
        if (
            path in primitive_paths
            or record.get("path") != str(path)
            or not path.is_file()
            or record.get("sha256") != _file_sha256(path)
            or record.get("sha256")
            != EXPECTED_PRIMITIVE_SHA256.get(relative.as_posix())
        ):
            raise ValueError("deployment primitive path/SHA drift")
        primitive_paths.add(path)
    file_paths: list[Path] = [manifest_path, state_path]
    destinations: set[Path] = set()
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "repo_relative_path",
            "source_path",
            "destination_path",
            "mode",
            "byte_count",
            "sha256",
        }:
            raise ValueError("deployment manifest file record drift")
        relative = _safe_relative_path(record["repo_relative_path"], label="file path")
        destination = _under(code_root / relative, code_root, label="destination")
        source = _under(frozen / relative, frozen, label="source path")
        if (
            record.get("destination_path") != str(destination)
            or record.get("source_path") != str(source)
            or destination in destinations
        ):
            raise ValueError("deployment manifest source/destination drift")
        destinations.add(destination)
        metadata = _reject_link_or_unsafe_mode(destination, label="code file")
        if metadata.st_uid != artifact_uid:
            raise ValueError("deployment code owner drift")
        if (
            record.get("mode") != stat.S_IMODE(metadata.st_mode)
            or record.get("byte_count") != metadata.st_size
            or record.get("sha256") != _file_sha256(destination)
        ):
            raise ValueError("deployment code byte or mode drift")
        normalized = relative.as_posix()
        override_sha = EXPECTED_CROSS_HOST_BUNDLE_SHA256.get(normalized)
        if override_sha is not None and record.get("sha256") != override_sha:
            raise ValueError("deployment cross-host override source/bundle drift")
        if source.exists() or source.is_symlink():
            _reject_link_or_unsafe_mode(source, label="source")
        if not _cross_host_source_mismatch_allowed(normalized):
            if not source.exists():
                raise ValueError("deployment non-owned frozen source is missing")
            if destination.read_bytes() != _bundle_content(source, normalized):
                raise ValueError("deployment code differs from available frozen source")
        _scan_for_secrets(destination.read_bytes(), label=relative.as_posix())
        file_paths.append(destination)
    if canonical_sha256(_release_records(files)) != release_pin:
        raise ValueError("deployment external release pin mismatch")
    expected_files = set(file_paths)
    actual_files = {
        path for path in deployment.rglob("*") if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("deployment contains an unlisted file or symlink")
    expected_dirs = _required_dirs(root, expected_files)
    actual_dirs = {path for path in deployment.rglob("*") if path.is_dir()} | {
        root,
        deployment,
    }
    if actual_dirs != expected_dirs:
        raise ValueError("deployment contains an unlisted directory")
    state_unsigned = dict(state_payload)
    state_sha = state_unsigned.pop("state_sha256", None)
    if (
        state_payload.get("schema_version") != STATE_SCHEMA
        or state_payload.get("status") != "canonical_pre_prepare_bundle"
        or state_sha != canonical_sha256(state_unsigned)
        or state_payload.get("owner") != owner
        or state_payload.get("owner_uid") != uid
        or state_payload.get("deployment_manifest_sha256")
        != manifest["manifest_unsigned_payload_sha256"]
        or state_payload.get("deployment_manifest_file_sha256")
        != _file_sha256(manifest_path)
        or state_payload.get("deployment_bundle_sha256")
        != manifest["deployment_bundle_sha256"]
        or state_payload.get("deployment_release_sha256") != release_pin
        or state_payload.get("actual_directory_manifest_sha256")
        != canonical_sha256(
            sorted(
                [MANIFEST_NAME, STATE_NAME]
                + [str(record["repo_relative_path"]) for record in files]
            )
        )
    ):
        raise ValueError("deployment state drift")
    return {
        "deployment_manifest_sha256": manifest["manifest_unsigned_payload_sha256"],
        "deployment_manifest_file_sha256": _file_sha256(manifest_path),
        "deployment_bundle_sha256": manifest["deployment_bundle_sha256"],
        "deployment_release_sha256": release_pin,
        "bundle_code_root": str(code_root),
        "frozen_repo_root": str(frozen),
        "owner": owner,
        "owner_uid": uid,
        "files": [str(path) for path in expected_files],
    }


def validate_deployment_bundle(
    v2_root: Path,
    *,
    frozen_repo_root: Path,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_owner: str | None = None,
    expected_owner_uid: int | None = None,
) -> dict[str, Any]:
    """Authenticate a deployment whose signed and filesystem owners agree."""
    return _validate_deployment_bundle(
        v2_root,
        frozen_repo_root=frozen_repo_root,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_owner=expected_owner,
        expected_owner_uid=expected_owner_uid,
    )


def rebind_transferred_deployment_owner(
    v2_root: Path,
    *,
    frozen_repo_root: Path,
    expected_source_release_sha256: str | None,
    expected_source_manifest_file_sha256: str | None,
    expected_source_owner_uid: int,
    target_owner: str | None = None,
    target_owner_uid: int | None = None,
) -> dict[str, Any]:
    """Re-sign only owner metadata after an authenticated cross-host transfer."""
    root = Path(v2_root).resolve(strict=True)
    frozen = Path(frozen_repo_root).resolve(strict=True)
    target_username, target_uid = _identity(target_owner, target_owner_uid)
    manifest_path = root / DEPLOYMENT_DIR / MANIFEST_NAME
    state_path = root / DEPLOYMENT_DIR / STATE_NAME
    manifest = _read_mapping(manifest_path, label="transfer manifest")
    source_owner = manifest.get("owner")
    if not isinstance(source_owner, str) or not source_owner:
        raise ValueError("deployment transfer source owner drift")
    _validate_deployment_bundle(
        root,
        frozen_repo_root=frozen,
        expected_release_sha256=expected_source_release_sha256,
        expected_manifest_file_sha256=expected_source_manifest_file_sha256,
        expected_owner=source_owner,
        expected_owner_uid=expected_source_owner_uid,
        filesystem_owner_uid=target_uid,
    )
    original_manifest = manifest_path.read_bytes()
    original_state = state_path.read_bytes()
    manifest["owner"] = target_username
    manifest["owner_uid"] = target_uid
    manifest_unsigned = _manifest_unsigned(manifest)
    manifest["manifest_unsigned_payload_sha256"] = canonical_sha256(
        manifest_unsigned
    )
    manifest["deployment_bundle_sha256"] = canonical_sha256(manifest_unsigned)
    manifest_bytes = _json_bytes(manifest)
    manifest_file_sha256 = _sha256_bytes(manifest_bytes)
    state = _read_mapping(state_path, label="transfer state")
    state["owner"] = target_username
    state["owner_uid"] = target_uid
    state["deployment_manifest_sha256"] = manifest[
        "manifest_unsigned_payload_sha256"
    ]
    state["deployment_manifest_file_sha256"] = manifest_file_sha256
    state["deployment_bundle_sha256"] = manifest["deployment_bundle_sha256"]
    state_unsigned = dict(state)
    state_unsigned.pop("state_sha256", None)
    state["state_sha256"] = canonical_sha256(state_unsigned)
    try:
        _atomic_replace_owned(manifest_path, manifest_bytes)
        _atomic_replace_owned(state_path, _json_bytes(state))
        verified = validate_deployment_bundle(
            root,
            frozen_repo_root=frozen,
            expected_release_sha256=expected_source_release_sha256,
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_owner=target_username,
            expected_owner_uid=target_uid,
        )
    except Exception:
        _atomic_replace_owned(manifest_path, original_manifest)
        _atomic_replace_owned(state_path, original_state)
        raise
    return {
        **verified,
        "source_owner": source_owner,
        "source_owner_uid": expected_source_owner_uid,
        "source_manifest_file_sha256": expected_source_manifest_file_sha256,
        "target_owner": target_username,
        "target_owner_uid": target_uid,
    }


def deployment_tree_files(v2_root: Path, verification: Mapping[str, Any]) -> set[Path]:
    """Return the authenticated deployment files for prepare exact-allowlist use."""
    root = Path(v2_root).resolve(strict=True)
    files = verification.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise ValueError("deployment verification file list drift")
    resolved = {Path(item).resolve(strict=True) for item in files}
    if not all(
        _under(path, root, label="verification file") == path for path in resolved
    ):
        raise ValueError("deployment verification escapes root")
    return resolved
