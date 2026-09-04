"""Convert an untrusted PyTorch checkpoint into a safetensors state dict.

This program is the *untrusted* half of the checkpoint boundary.  Loading a
participant supplied ``.pt`` file can execute memory-unsafe deserialization
code in affected PyTorch releases, even with ``weights_only=True``.  Therefore
this file must run as the participant UID (or in a disposable, networkless
container) and must never be given leaderboard secrets or private test data.

The trusted scorer consumes only the resulting ``.safetensors`` file.  It
never calls ``torch.load`` on a participant artifact.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import resource
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save


DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TENSORS = 100_000
PR_SET_NO_NEW_PRIVS = 38


def _lock_process_limits(max_bytes: int, cpu_seconds: int) -> None:
    """Install irreversible best-effort limits before deserializing input."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if cpu_seconds > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    # A failure here is non-fatal on non-Linux development hosts. Production
    # additionally uses Docker's no-new-privileges and capability drop.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except (AttributeError, OSError):
        pass


def _plain_state_dict(raw: object, max_bytes: int, max_tensors: int) -> dict[str, torch.Tensor]:
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint must contain a state_dict mapping")
    state = raw.get("model", raw)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model must be a non-empty state_dict mapping")
    if len(state) > max_tensors:
        raise ValueError(f"checkpoint has more than {max_tensors} tensors")

    converted: dict[str, torch.Tensor] = {}
    total_bytes = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError("state_dict keys must be non-empty strings")
        if type(tensor) is not torch.Tensor or tensor.layout is not torch.strided:
            raise ValueError(f"state_dict value {key!r} must be a dense plain tensor")
        if tensor.device.type == "meta":
            raise ValueError(f"state_dict value {key!r} must have materialized storage")
        tensor_bytes = tensor.numel() * tensor.element_size()
        total_bytes += tensor_bytes
        if total_bytes > max_bytes:
            raise ValueError(f"state_dict tensors exceed the {max_bytes}-byte limit")
        # Materialize ordinary, independent CPU tensors. This also prevents
        # safetensors shared-storage rejection from depending on pickle layout.
        converted[key] = tensor.detach().to(device="cpu").contiguous().clone()
    return converted


def _write_all(file_descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("could not write converted checkpoint")
        view = view[written:]
    os.fsync(file_descriptor)


def convert(args: argparse.Namespace) -> None:
    if args.max_bytes <= 0 or args.max_tensors <= 0:
        raise ValueError("conversion limits must be positive")
    _lock_process_limits(args.max_bytes, args.cpu_seconds)

    # Deliberately confined to this untrusted process. Do not move this load
    # into the API or scorer, even if a future call site uses weights_only.
    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    tensors = _plain_state_dict(raw, args.max_bytes, args.max_tensors)
    serialized = save(tensors)
    if not serialized or len(serialized) > args.max_bytes:
        raise ValueError("converted checkpoint exceeds the output size limit")

    if args.output_fd is not None:
        _write_all(args.output_fd, serialized)
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    output_fd = os.open(args.output, flags, 0o600)
    try:
        _write_all(output_fd, serialized)
    finally:
        os.close(output_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--output-fd", type=int)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MAX_TENSORS)
    parser.add_argument("--cpu-seconds", type=int, default=300)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
