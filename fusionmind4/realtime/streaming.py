"""
Streaming Plasma Interface
============================

Real-time data ingestion from tokamak diagnostic systems.

Supports:
  1. Live mode:   ZeroMQ/TCP stream from PCS (Plasma Control System)
  2. Replay mode: Historical shot data at configurable speed
  3. FAIR-MAST mode: S3 zarr streaming for validation

Data flow:
  Diagnostic ADC → PCS → ZMQ pub → StreamingPlasmaInterface
                                         │
                                    ┌─────┘
                                    ▼
                              PlasmaSnapshot
                                    │
                              ┌─────┘
                              ▼
                        DualModePredictor
                              │
                        ┌─────┘
                        ▼
                  ControlBridge → Actuator commands

Protocol:
  Each diagnostic message is a dict:
    {
      'timestamp_s': float,       # seconds since SOD
      'shot_id': int,
      'signals': {
        'betaN': float,
        'ne': float,
        ...
      }
    }

Patent Families: PF1 (CPDE), PF2 (CPC)
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
from .predictor import PlasmaSnapshot


@dataclass
class StreamConfig:
    """Configuration for streaming interface."""
    mode: str = 'replay'           # 'live', 'replay', 'fair_mast'
    replay_speed: float = 1.0      # 1x = real-time, 10x = 10x faster
    buffer_size: int = 1000        # Rolling buffer size
    callback_interval_ms: float = 1.0  # Callback frequency
    zmq_address: str = 'tcp://localhost:5555'  # For live mode
    fair_mast_shots: Optional[List[int]] = None


class StreamingPlasmaInterface:
    """Interface for streaming plasma data into FusionMind pipeline.

    Handles data ingestion, buffering, time-alignment, and callback
    dispatch for real-time operation.
    """

    def __init__(self, config: StreamConfig,
                 var_names: List[str]):
        self.config = config
        self.var_names = list(var_names)
        self._buffer: deque = deque(maxlen=config.buffer_size)
        self._callbacks: List[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot_count = 0

        # Performance tracking
        self._ingest_latencies: List[float] = []

    def register_callback(self, callback: Callable[[PlasmaSnapshot], None]):
        """Register a callback to receive each new PlasmaSnapshot."""
        self._callbacks.append(callback)

    def start(self):
        """Start streaming in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self):
        """Main streaming loop."""
        if self.config.mode == 'replay':
            self._run_replay()
        elif self.config.mode == 'live':
            self._run_live()
        elif self.config.mode == 'fair_mast':
            self._run_fair_mast()

    def _run_replay(self):
        """Replay synthetic or historical data."""
        rng = np.random.RandomState(42)
        t = 0.0
        dt = self.config.callback_interval_ms / 1000.0

        while self._running:
            t0 = time.perf_counter()

            # Generate synthetic plasma state
            values = self._generate_synthetic_state(t, rng)
            snapshot = PlasmaSnapshot(
                values=values,
                timestamp_s=t,
                shot_id=0,
            )
            self._dispatch(snapshot)
            t += dt * self.config.replay_speed

            # Sleep to maintain real-time pacing
            elapsed = time.perf_counter() - t0
            sleep_time = (dt / self.config.replay_speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_live(self):
        """Receive live data via ZeroMQ (requires zmq installed)."""
        try:
            import zmq
        except ImportError:
            raise RuntimeError(
                "zmq required for live mode: pip install pyzmq"
            )

        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect(self.config.zmq_address)
        subscriber.subscribe(b'plasma')

        while self._running:
            try:
                topic, msg = subscriber.recv_multipart(flags=zmq.NOBLOCK)
                import json
                data = json.loads(msg.decode())
                snapshot = PlasmaSnapshot(
                    values=data.get('signals', {}),
                    timestamp_s=data.get('timestamp_s', 0.0),
                    shot_id=data.get('shot_id', 0),
                )
                self._dispatch(snapshot)
            except zmq.Again:
                time.sleep(0.0001)  # 100 μs poll

        subscriber.close()
        context.destroy()

    def _run_fair_mast(self):
        """Stream from FAIR-MAST S3 zarr data."""
        try:
            import s3fs
            import zarr
        except ImportError:
            raise RuntimeError(
                "s3fs and zarr required for FAIR-MAST mode"
            )

        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'endpoint_url': 'https://s3.echo.stfc.ac.uk'}
        )

        shots = self.config.fair_mast_shots or [27880, 27881, 27882]

        for shot_id in shots:
            if not self._running:
                break

            path = f"mast/level1/shots/{shot_id}.zarr"
            try:
                store = s3fs.S3Map(root=path, s3=fs, check=False)
                root = zarr.open(store, mode='r')
            except Exception:
                continue

            # Extract EFIT signals
            if 'efm' not in root:
                continue

            efm = root['efm']
            keys_map = {
                'betan': 'βN', 'betap': 'βp', 'q_95': 'q95',
                'q_axis': 'q_axis', 'li': 'li', 'elongation': 'κ',
            }

            # Get time array
            if 'all_times' in efm:
                times = np.array(efm['all_times'])
            else:
                continue

            # Load arrays
            arrays = {}
            for zkey, label in keys_map.items():
                if zkey in efm:
                    try:
                        arrays[label] = np.array(efm[zkey])
                    except Exception:
                        pass

            # Stream each timepoint
            dt = self.config.callback_interval_ms / 1000.0
            for i in range(0, len(times), max(1, int(1.0 / self.config.replay_speed))):
                if not self._running:
                    break

                values = {}
                for label, arr in arrays.items():
                    if i < len(arr):
                        val = float(arr[i]) if np.isscalar(arr[i]) else float(arr[i].flat[0])
                        if np.isfinite(val):
                            values[label] = val

                if values:
                    snapshot = PlasmaSnapshot(
                        values=values,
                        timestamp_s=float(times[i]),
                        shot_id=shot_id,
                    )
                    self._dispatch(snapshot)

                sleep_time = dt / self.config.replay_speed
                if sleep_time > 0.0001:
                    time.sleep(sleep_time)

    def _dispatch(self, snapshot: PlasmaSnapshot):
        """Dispatch snapshot to all callbacks and buffer."""
        t0 = time.perf_counter()
        self._buffer.append(snapshot)
        self._snapshot_count += 1
        for cb in self._callbacks:
            cb(snapshot)
        latency_us = (time.perf_counter() - t0) * 1e6
        self._ingest_latencies.append(latency_us)
        if len(self._ingest_latencies) > 10000:
            self._ingest_latencies = self._ingest_latencies[-5000:]

    def _generate_synthetic_state(self, t: float,
                                  rng: np.random.RandomState) -> Dict[str, float]:
        """Generate synthetic plasma state for replay mode."""
        noise = rng.randn(len(self.var_names)) * 0.05

        # Base profiles (MAST-like spherical tokamak)
        state: Dict[str, float] = {}

        # Phase of discharge
        phase = np.clip(t / 0.5, 0, 1)  # 0-1 over 500ms
        heating = 2.0 + 1.5 * np.sin(2 * np.pi * t / 0.3)

        defaults = {
            'P_NBI': max(0, heating + noise[0] if len(noise) > 0 else heating),
            'P_ECRH': max(0, 0.5 + 0.3 * phase + (noise[1] if len(noise) > 1 else 0)),
            'gas_puff': max(0, 1.0 + 0.5 * np.sin(t * 5) + (noise[2] if len(noise) > 2 else 0)),
            'Ip': 0.8 + 0.2 * phase,
            'ne': 3.0 + 1.0 * phase + (noise[3] if len(noise) > 3 else 0),
            'ne_core': 3.5 + 1.2 * phase,
            'Te': 1.0 + 2.0 * heating / 3.5,
            'Ti': 0.8 + 1.5 * heating / 3.5,
            'q': 6.0 - 1.0 * phase,
            'q95': 5.5 - 0.8 * phase,
            'q_axis': 1.0 + 0.2 * np.sin(t * 3),
            'betaN': 0.5 + 0.8 * heating / 3.5,
            'βN': 0.5 + 0.8 * heating / 3.5,
            'βp': 0.2 + 0.1 * heating / 3.5,
            'rotation': 1e4 * heating / 3.5,
            'P_rad': 0.3 * heating,
            'W_stored': 0.1 + 0.15 * heating / 3.5,
            'MHD_amp': max(0, 0.01 + 0.005 * np.sin(t * 20)),
            'n_imp': 0.01 + 0.005 * t,
            'li': 1.0 + 0.1 * np.sin(t * 4),
            'κ': 1.8 + 0.02 * np.sin(t * 2),
            'D_alpha': 0.5 + 0.2 * phase,
        }

        for v in self.var_names:
            if v in defaults:
                state[v] = defaults[v]
            else:
                state[v] = float(rng.randn())

        return state

    def get_buffer_as_array(self) -> np.ndarray:
        """Return buffered data as (N, n_vars) array."""
        if not self._buffer:
            return np.empty((0, len(self.var_names)))
        rows = []
        for snap in self._buffer:
            row = snap.to_array(self.var_names)
            rows.append(row)
        return np.array(rows)

    def get_statistics(self) -> Dict:
        """Streaming performance statistics."""
        if not self._ingest_latencies:
            return {}
        lats = self._ingest_latencies
        return {
            'snapshots_ingested': self._snapshot_count,
            'buffer_size': len(self._buffer),
            'ingest_latency_mean_us': float(np.mean(lats)),
            'ingest_latency_p99_us': float(np.percentile(lats, 99)),
            'mode': self.config.mode,
        }
