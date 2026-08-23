"""Real 3-DISTINCT-machine N>2 topology proof - the one gap test_n_gt_2.py
couldn't close (it only runs all 3 ranks as local processes on one
machine's loopback). Here each rank is a genuinely separate real
machine: rank 0 = local sandbox, rank 1 = akun5 (the pipeline's middle
stage - the one that must hold TWO real concurrent cross-machine QUIC
connections at once, one to each of the other two, on two real
different network paths), rank 2 = a third real machine.

Topology: rank 0 -> rank 1 -> rank 2, same shape as test_n_gt_2.py's
loopback version, just for real over the internet instead of localhost.

Usage: python3 n3_cross_machine_rank.py <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import quic_dist
import torch.distributed as dist

if __import__("os").environ.get("QUIC_DIST_FAULTHANDLER"):
    # py-spy needs ptrace, which this container's seccomp profile blocks
    # outright (confirmed: `cap_sys_ptrace` denied, an active seccomp
    # filter) - faulthandler dumps all thread stacks from IN-PROCESS via
    # a plain signal, no ptrace needed, so it still works here.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)

rank = int(sys.argv[1])
signaling_url = sys.argv[2]
job_id = sys.argv[3] if len(sys.argv) > 3 else "n3_cross"

quic_dist.init_process_group(
    signaling_url=signaling_url, rank=rank, world_size=3, job_id=job_id, timeout=timedelta(seconds=120)
)
print(f"[rank {rank}] process group ready", flush=True)

try:
    if rank == 0:
        tensor = torch.arange(100, dtype=torch.float32)
        dist.send(tensor, dst=1, tag=0)
        print(f"[rank 0] sent to rank 1", flush=True)
    elif rank == 1:
        recv_buf = torch.zeros(100, dtype=torch.float32)
        dist.recv(recv_buf, src=0, tag=0)
        expected = torch.arange(100, dtype=torch.float32)
        assert torch.equal(recv_buf, expected), "rank1: received from rank0 MISMATCH"
        print(f"[rank 1] received from rank 0 OK (byte-exact)", flush=True)

        pg = dist.distributed_c10d._get_default_group()
        assert 0 in pg._peers and 1 not in pg._peers, f"rank1 peer state wrong before 2nd connect: {list(pg._peers.keys())}"
        print(f"[rank 1] peers before 2nd send: {list(pg._peers.keys())} (expected [0])", flush=True)

        dist.send(recv_buf * 2, dst=2, tag=0)
        print(f"[rank 1] sent to rank 2", flush=True)
        assert len(pg._peers) == 2, f"rank1 should hold 2 concurrent REAL cross-machine connections, has {len(pg._peers)}"
        print(f"[rank 1] CONFIRMED: holding {len(pg._peers)} concurrent real cross-machine QUIC connections: {list(pg._peers.keys())}", flush=True)
    else:  # rank == 2
        recv_buf = torch.zeros(100, dtype=torch.float32)
        dist.recv(recv_buf, src=1, tag=0)
        expected = torch.arange(100, dtype=torch.float32) * 2
        assert torch.equal(recv_buf, expected), "rank2: received from rank1 MISMATCH"
        print(f"[rank 2] received from rank 1 OK (byte-exact)", flush=True)

    dist.barrier()
    print(f"[rank {rank}] real 3-way cross-machine barrier PASSED", flush=True)
    print(f"[rank {rank}] RESULT: OK", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[rank {rank}] RESULT: FAILED: {exc!r}", flush=True)
    raise
finally:
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)
