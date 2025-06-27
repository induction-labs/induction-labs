from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
from synapse.actions.models import Action, MouseMove


@dataclass
class Cubic:
    # specify a cubic that passes through (0, 0) and (1, a)
    # with tangent at (0, 0) being m and tangent at (1, m) being n
    # can convert to a polynomial with
    # f(x) = (m+n-2a)x^3 + (3a-m-2n)x^2 + nx
    m: float
    n: float
    a: float

    def coeffs(self) -> tuple[float, float, float, float]:
        """Return the four polynomial coefficients (c3, c2, c1, c0)."""
        c3 = self.m + self.n - 2 * self.a
        c2 = 3 * self.a - 2 * self.m - self.n
        c1 = self.m
        c0 = 0.0
        return c3, c2, c1, c0

    def to_ndarray(self) -> np.ndarray:
        """Return the coefficients as a numpy array."""
        return np.array([self.m, self.n, self.a])

    def __call__(self, x):
        """Evaluate the cubic at x (scalar or array)."""
        c3, c2, c1, _ = self.coeffs()
        return ((c3 * x + c2) * x + c1) * x  # Horner form


def create_cubic_from_path(path: np.ndarray) -> Cubic:
    # path is shape [n, 2]; eg [(x1, y1), (x2, y2), ...]
    # returns a cubic that passes through (0, 0)
    # fit the best cubic to the path
    # points should be sorted

    # normalize the time to [0, 1]
    assert path.ndim == 2 and path.shape[1] == 2, "Path must be [n, 2]"

    x_vals = path[:, 0]
    assert np.all(np.diff(x_vals) >= 0), "Path x-coordinates must be sorted"

    # Translate so the first point is exactly (0, 0)
    path_shifted = path - path[0]

    # Independent variable: either the given x coordinates (if they vary)
    # or a uniform parameter if all x's are equal.
    x_raw = path_shifted[:, 0]
    x_raw = (x_raw - x_raw[0]) / (x_raw[-1] - x_raw[0])

    y = path_shifted[:, 1]
    a_target = y[-1]  # y-value at x = 1

    # design matrix for coefficients [c3, c2, c1]
    X = np.column_stack((x_raw**3, x_raw**2, x_raw))

    # ------------------------------------------------------------------
    # Solve min ||X c - y||²  subject to  L c = b  where L = [1 1 1]
    # Use the KKT system:
    #       [XᵀX  Lᵀ] [c] = [Xᵀy]
    #       [ L    0 ] [λ]   [ b  ]
    # ------------------------------------------------------------------
    XtX = X.T @ X
    XtY = X.T @ y
    L = np.array([[1.0, 1.0, 1.0]])
    b = np.array([a_target])

    K = np.block([[XtX, L.T], [L, np.zeros((1, 1))]])
    rhs = np.concatenate([XtY, b])

    sol = np.linalg.solve(K, rhs)
    c3, c2, c1 = sol[:3]  # constrained LS solution

    # convert to (m, n, a)
    m = c1
    n = 3 * c3 + 2 * c2 + c1
    a = a_target

    return Cubic(m, n, a)


def get_all_action_logs(bucket_path: str):
    """Process files from GCS bucket."""
    fs = gcsfs.GCSFileSystem()

    # Remove gs:// prefix
    bucket_path = bucket_path[5:]

    try:
        files = fs.glob(f"{bucket_path}/action_capture_*.jsonl")
    except Exception as e:
        print(f"Error listing files from {bucket_path}: {e}")
        return

    all_actions = []

    for file_path in files:
        filename = file_path.split("/")[-1]

        try:
            with fs.open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_actions.append(Action(**json.loads(line)))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_actions


def lerp_coordinate(t0: float, v0: float, t1: float, v1: float, t: float) -> float:
    """Linear interpolation helper (no clamping)."""
    if t1 == t0:
        return v0  # degenerate, shouldn't happen but stay safe
    return v0 + (v1 - v0) * ((t - t0) / (t1 - t0))


def fill_gaps(data: list[Action], step=0.033, fields=("x", "y")):
    new_data = []
    for i in range(len(data) - 1):
        curr = data[i]
        nxt = data[i + 1]
        new_data.append(curr)
        dt = nxt.timestamp - curr.timestamp
        if dt > step:
            n_steps = int(dt // step)
            for j in range(1, n_steps + 1):
                t = curr.timestamp + j * step
                if t < nxt.timestamp:
                    ratio = (t - curr.timestamp) / dt
                    current_data = Action(action=copy.copy(curr.action), timestamp=t)
                    for field in fields:
                        current_data.action.__setattr__(
                            field,
                            curr.action.__getattribute__(field)
                            + ratio
                            * (
                                nxt.action.__getattribute__(field)
                                - curr.action.__getattribute__(field)
                            ),
                        )
                    new_data.append(current_data)
    new_data.append(data[-1])
    return new_data


def process_continuous_actions(
    timestamps: np.ndarray,
    actions: list[Action],
    original_screen_size: tuple[int, int],
    action_factory: Callable[[float, float], Action] = lambda x, y: MouseMove(x=x, y=y),
) -> tuple[list[Cubic], list[Cubic]]:
    timestamps = timestamps[1:]

    current_mouse_idx = 0

    poly_xs = []
    poly_ys = []

    for i, t_hi in enumerate(timestamps):
        current_mouse_move_buffer = []

        previous_mouse_move_idx = current_mouse_idx - 1
        while (
            current_mouse_idx < len(actions)
            and actions[current_mouse_idx].timestamp <= t_hi
        ):
            current_mouse_move_buffer.append(actions[current_mouse_idx])
            current_mouse_idx += 1

        next_mouse_move_idx = current_mouse_idx

        if current_mouse_idx >= len(actions):
            break

        # add two points at timestamps[i-1] and timestamps[1+1] exactly, by lerping
        # previous action and current action [0] for timestamps[i-1]
        # next action and current action [-1] for timestamps[i+1]
        if (
            i > 0
            and previous_mouse_move_idx >= 0
            and previous_mouse_move_idx + 1 < len(actions)
        ):
            t_lo = timestamps[i - 1] if i > 0 else None
            previous_move = actions[previous_mouse_move_idx]
            previous_move_plus_one = actions[previous_mouse_move_idx + 1]
            x = lerp_coordinate(
                previous_move.timestamp,
                previous_move.action.x,
                previous_move_plus_one.timestamp,
                previous_move_plus_one.action.x,
                t_lo,
            )
            y = lerp_coordinate(
                previous_move.timestamp,
                previous_move.action.y,
                previous_move_plus_one.timestamp,
                previous_move_plus_one.action.y,
                t_lo,
            )

            first_event = Action(
                action=action_factory(float(x), float(y)), timestamp=t_lo
            )
            current_mouse_move_buffer.insert(0, first_event)

        if next_mouse_move_idx < len(actions):
            next_move = actions[next_mouse_move_idx]
            next_move_minus_one = actions[next_mouse_move_idx - 1]
            x = lerp_coordinate(
                next_move_minus_one.timestamp,
                next_move_minus_one.action.x,
                next_move.timestamp,
                next_move.action.x,
                t_hi,
            )
            y = lerp_coordinate(
                next_move_minus_one.timestamp,
                next_move_minus_one.action.y,
                next_move.timestamp,
                next_move.action.y,
                t_hi,
            )

            last_event = Action(
                action=action_factory(float(x), float(y)), timestamp=t_hi
            )
            current_mouse_move_buffer.append(last_event)

        if len(current_mouse_move_buffer) == 2:
            # put timestamp between and lerp
            in_between_timestamp = (
                current_mouse_move_buffer[0].timestamp
                + current_mouse_move_buffer[1].timestamp
            ) / 2
            current_mouse_move_buffer.insert(
                1,
                Action(
                    action=action_factory(
                        (
                            current_mouse_move_buffer[0].action.x
                            + current_mouse_move_buffer[1].action.x
                        )
                        / 2,
                        (
                            current_mouse_move_buffer[0].action.y
                            + current_mouse_move_buffer[1].action.y
                        )
                        / 2,
                    ),
                    timestamp=in_between_timestamp,
                ),
            )

        mouse_x = np.array(
            [
                (action.timestamp, action.action.x / original_screen_size[0])
                for action in current_mouse_move_buffer
            ]
        )
        mouse_y = np.array(
            [
                (action.timestamp, action.action.y / original_screen_size[1])
                for action in current_mouse_move_buffer
            ]
        )

        poly_x = create_cubic_from_path(mouse_x)
        poly_y = create_cubic_from_path(mouse_y)
        poly_xs.append(poly_x)
        poly_ys.append(poly_y)

    return poly_xs, poly_ys


def plot_timestamp_comparison(
    timestamps, actions, poly_xs, poly_ys, original_screen_size
):
    """Plot original timestamps vs polynomial modeled ones with continuous polynomials"""

    # Filter mouse move actions

    # Create subplots for x and y coordinates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot all original data first
    orig_times_all = [action.timestamp for action in actions]
    orig_x_all = [action.action.x for action in actions]
    orig_y_all = [action.action.y for action in actions]

    ax1.plot(
        orig_times_all,
        orig_x_all,
        "o-",
        alpha=0.6,
        markersize=2,
        linewidth=1,
        color="blue",
        label="Original X",
    )
    ax2.plot(
        orig_times_all,
        orig_y_all,
        "o-",
        alpha=0.6,
        markersize=2,
        linewidth=1,
        color="blue",
        label="Original Y",
    )

    # Now build the continuous polynomial curve
    all_poly_times = []
    all_poly_x = []
    all_poly_y = []

    # Initialize previous last x and y coordinates
    previous_last_x = orig_x_all[0]
    previous_last_y = orig_y_all[0]
    for i, (poly_x, poly_y) in enumerate(zip(poly_xs, poly_ys, strict=False)):
        # Generate dense x values for the polynomial
        x_dense = np.linspace(0, 1, 400)
        y_cubic_of_xcoord = poly_x(x_dense)
        y_cubic_of_ycoord = poly_y(x_dense)
        all_poly_times.extend(
            x_dense * (timestamps[i + 1] - timestamps[i]) + timestamps[i]
        )
        all_poly_x.extend(y_cubic_of_xcoord * original_screen_size[0] + previous_last_x)
        all_poly_y.extend(y_cubic_of_ycoord * original_screen_size[1] + previous_last_y)

        previous_last_x = all_poly_x[-1]
        previous_last_y = all_poly_y[-1]

    # Plot continuous polynomial curve
    if all_poly_times:
        ax1.plot(
            all_poly_times,
            all_poly_x,
            "-",
            color="red",
            linewidth=2,
            label="Polynomial Model X",
            alpha=0.8,
        )
        ax2.plot(
            all_poly_times,
            all_poly_y,
            "-",
            color="red",
            linewidth=2,
            label="Polynomial Model Y",
            alpha=0.8,
        )

    # draw lines at each timestamp
    for t in timestamps:
        ax1.axvline(t, color="gray", linestyle="--", alpha=0.3)
        ax2.axvline(t, color="gray", linestyle="--", alpha=0.3)

    # Format plots
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("X Coordinate (pixels)")
    ax1.set_title("X Coordinate: Original vs Polynomial Model")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Y Coordinate (pixels)")
    ax2.set_title("Y Coordinate: Original vs Polynomial Model")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate the comparison plot
    logs = get_all_action_logs(
        "gs://induction-labs-data-ext/action_capture/jonathan/2025-06-26_164012_TG1MZ"
    )
    timestamps = np.arange(logs[0].timestamp - 0.2, logs[-1].timestamp, 0.5)
    # add some noise to timestamps
    timestamps += np.random.uniform(-0.05, 0.05, size=timestamps.shape)

    # steps to get
    mouse_move_actions = [
        action for action in logs if isinstance(action.action, MouseMove)
    ]
    for action in mouse_move_actions:
        action.timestamp += 0.05

    filled_actions = fill_gaps(mouse_move_actions, step=0.01)
    screen_size = (1440, 900)
    mouse_x_cubic, mouse_y_cubic = process_continuous_actions(
        timestamps, filled_actions, screen_size
    )

    plot_timestamp_comparison(
        timestamps, filled_actions, mouse_x_cubic, mouse_y_cubic, screen_size
    )
