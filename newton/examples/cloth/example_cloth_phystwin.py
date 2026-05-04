# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example PhysTwin Cloth
#
# This example loads a PhysTwin `.npz` export and runs it as a mass-spring
# particle simulation in Newton. If the export includes controller anchor
# vertices, they are added as kinematic particles and driven by the exported
# controller trajectory.
#
# The PhysTwin export used in this workspace contains particles and springs,
# but no triangle faces, so the viewer renders a spring-particle cloth rather
# than a shaded surface mesh.
#
# Command: uv run -m newton.examples cloth_phystwin
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ_PATH = REPO_ROOT / "physics_export" / "physics_params.npz"


@wp.kernel
def set_kinematic_particles(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    start_index: int,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_q[start_index + tid] = positions[tid]
    particle_qd[start_index + tid] = velocities[tid]


@wp.kernel
def apply_particle_drag(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    drag_damping: float,
    dt: float,
):
    tid = wp.tid()
    if particle_mass[tid] > 0.0:
        particle_qd[tid] *= wp.max(0.0, 1.0 - drag_damping * dt)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.show_springs = args.show_springs
        self.stiffness_mode = args.stiffness_mode

        self.npz_path = Path(args.npz_path).expanduser().resolve()
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"PhysTwin export not found: {self.npz_path}\n"
                "Pass a valid file with `--npz-path /path/to/physics_params.npz`."
            )

        self._load_export_data()

        self.original_substeps = max(1, self.original_substeps)
        self.controller_frame_dt = self.export_dt * self.original_substeps
        self.sim_substeps = args.substeps if args.substeps is not None else self.original_substeps
        self.sim_substeps = max(1, self.sim_substeps)
        self.sim_dt = self.controller_frame_dt / self.sim_substeps
        self.frame_dt = self.controller_frame_dt

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for pos, mass in zip(self.object_positions, self.object_masses):
            builder.add_particle(
                pos=wp.vec3(*map(float, pos)),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=float(mass),
                radius=self.particle_radius,
            )

        for pos in self.controller_positions_0:
            builder.add_particle(
                pos=wp.vec3(*map(float, pos)),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.0,
                radius=self.particle_radius,
            )

        for (i, j), ke, kd in zip(self.spring_indices, self.spring_stiffness, self.spring_damping):
            builder.add_spring(int(i), int(j), float(ke), float(kd), 0.0)

        self.model = builder.finalize()
        self.model.spring_rest_length.assign(self.spring_rest_lengths)
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_kf = 1.0e2
        self.model.soft_contact_mu = self.contact_friction
        self.model.soft_contact_restitution = self.contact_restitution

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.initial_positions = np.concatenate((self.object_positions, self.controller_positions_0), axis=0)
        self.initial_velocities = np.zeros_like(self.initial_positions)

        self.current_frame_index = 0
        self.current_controller_positions = None
        self.current_controller_velocities = None
        if self.controller_particle_count:
            self.current_controller_positions = wp.array(
                self.controller_positions_0,
                dtype=wp.vec3,
                device=self.model.device,
            )
            self.current_controller_velocities = wp.zeros(
                self.controller_particle_count,
                dtype=wp.vec3,
                device=self.model.device,
            )
        self._reset_states()

        self.viewer.set_model(self.model)
        self._configure_viewer()
        self.capture()

        print(
            "Loaded PhysTwin cloth export:",
            f"{self.object_particle_count} object particles,",
            f"{self.controller_particle_count} controller particles,",
            f"{self.spring_count} springs,",
            f"stiffness_mode={self.resolved_stiffness_mode},",
            f"sim_dt={self.sim_dt:.6e}s,",
            f"substeps={self.sim_substeps}",
        )

    def _load_export_data(self) -> None:
        with np.load(self.npz_path) as data:
            self.object_positions = np.array(data["object_vertices_0"], dtype=np.float32)
            self.object_masses = np.array(data["masses"], dtype=np.float32)
            self.spring_indices = np.array(data["springs"], dtype=np.int32)
            self.spring_rest_lengths = np.array(data["rest_lengths"], dtype=np.float32)
            self.spring_y = np.array(data["spring_Y"], dtype=np.float32)
            self.controller_positions_0 = np.array(
                data["controller_vertices_0"] if "controller_vertices_0" in data else np.empty((0, 3)),
                dtype=np.float32,
            )
            self.controller_trajectory = np.array(
                data["controller_trajectory"] if "controller_trajectory" in data else np.empty((0, 0, 3)),
                dtype=np.float32,
            )
            self.export_dt = float(data["dt"]) if "dt" in data else 1.0 / 30.0
            self.original_substeps = int(data["num_substeps"]) if "num_substeps" in data else 1
            self.particle_radius = float(data["collision_dist"]) if "collision_dist" in data else 0.02
            self.drag_damping = float(data["drag_damping"]) if "drag_damping" in data else 0.0
            self.export_dashpot_damping = float(data["dashpot_damping"]) if "dashpot_damping" in data else 0.0
            self.contact_friction = float(data["collide_fric"]) if "collide_fric" in data else 0.3
            self.contact_restitution = float(data["collide_elas"]) if "collide_elas" in data else 0.0

        if self.object_positions.ndim != 2 or self.object_positions.shape[1] != 3:
            raise ValueError("`object_vertices_0` must have shape (N, 3).")
        if self.object_masses.shape[0] != self.object_positions.shape[0]:
            raise ValueError("`masses` length must match `object_vertices_0`.")
        if self.spring_indices.ndim != 2 or self.spring_indices.shape[1] != 2:
            raise ValueError("`springs` must have shape (S, 2).")
        if self.spring_rest_lengths.shape[0] != self.spring_indices.shape[0]:
            raise ValueError("`rest_lengths` length must match `springs`.")
        if self.spring_y.shape[0] != self.spring_indices.shape[0]:
            raise ValueError("`spring_Y` length must match `springs`.")
        if self.controller_positions_0.ndim != 2 or self.controller_positions_0.shape[1] != 3:
            raise ValueError("`controller_vertices_0` must have shape (M, 3) when present.")
        if self.controller_trajectory.size > 0:
            if self.controller_trajectory.ndim != 3 or self.controller_trajectory.shape[2] != 3:
                raise ValueError("`controller_trajectory` must have shape (T, M, 3).")
            if self.controller_trajectory.shape[1] != self.controller_positions_0.shape[0]:
                raise ValueError(
                    "`controller_trajectory` control-point count must match `controller_vertices_0`."
                )

        self.object_particle_count = self.object_positions.shape[0]
        self.controller_particle_count = self.controller_positions_0.shape[0]
        self.controller_start_index = self.object_particle_count
        self.total_particle_count = self.object_particle_count + self.controller_particle_count
        self.spring_count = self.spring_indices.shape[0]

        max_spring_index = int(self.spring_indices.max()) if self.spring_count else -1
        if max_spring_index >= self.total_particle_count:
            raise ValueError(
                "Spring topology references particle indices beyond the available object and controller particles. "
                "This export likely needs `controller_vertices_0` or a different topology mapping."
            )

        self.spring_stiffness, self.spring_damping, self.resolved_stiffness_mode = self._decode_springs()

    def _decode_springs(self) -> tuple[np.ndarray, np.ndarray, str]:
        if self.stiffness_mode == "direct":
            spring_stiffness = self.spring_y.copy()
            mode = "direct"
        elif self.stiffness_mode == "exp":
            spring_stiffness = np.exp(self.spring_y).astype(np.float32)
            mode = "exp"
        else:
            # PhysTwin exports in this workspace store already-scaled stiffness
            # values (~5e3), while earlier test fixtures stored log stiffness.
            if float(np.max(self.spring_y)) > 80.0:
                spring_stiffness = self.spring_y.copy()
                mode = "direct"
            else:
                spring_stiffness = np.exp(self.spring_y).astype(np.float32)
                mode = "exp"

        if mode == "direct":
            spring_damping = np.full(
                self.spring_count,
                fill_value=self.export_dashpot_damping,
                dtype=np.float32,
            )
        else:
            spring_damping = (0.1 * spring_stiffness).astype(np.float32)

        return spring_stiffness.astype(np.float32), spring_damping, mode

    def _configure_viewer(self) -> None:
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False

        p_min = np.min(self.object_positions, axis=0)
        p_max = np.max(self.object_positions, axis=0)
        center = 0.5 * (p_min + p_max)
        extent = max(float(np.max(p_max - p_min)), 0.2)
        distance = 2.2 * extent

        self.viewer.set_camera(
            pos=wp.vec3(
                float(center[0] + distance),
                float(center[1] - 1.1 * distance),
                float(center[2] + 0.9 * distance),
            ),
            pitch=-20.0,
            yaw=140.0,
        )

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self._reset_states()
        else:
            self.graph = None

    def _reset_states(self) -> None:
        self.current_frame_index = 0
        if self.controller_particle_count:
            self._set_controller_frame(0)
        self.state_0.particle_q.assign(self.initial_positions)
        self.state_0.particle_qd.assign(self.initial_velocities)
        self.state_1.particle_q.assign(self.initial_positions)
        self.state_1.particle_qd.assign(self.initial_velocities)

    def _set_controller_frame(self, frame_index: int) -> None:
        if self.controller_particle_count == 0:
            return

        if self.controller_trajectory.shape[0] == 0:
            positions = self.controller_positions_0
            velocities = np.zeros_like(positions)
        else:
            frame_index = min(frame_index, self.controller_trajectory.shape[0] - 1)
            positions = self.controller_trajectory[frame_index]
            if self.controller_trajectory.shape[0] == 1:
                velocities = np.zeros_like(positions)
            elif frame_index == 0:
                velocities = (self.controller_trajectory[1] - self.controller_trajectory[0]) / self.frame_dt
            else:
                velocities = (self.controller_trajectory[frame_index] - self.controller_trajectory[frame_index - 1]) / self.frame_dt

        self.current_controller_positions.assign(positions.astype(np.float32))
        self.current_controller_velocities.assign(velocities.astype(np.float32))

    def _apply_controller_targets(self, state: newton.State) -> None:
        if self.controller_particle_count == 0:
            return

        wp.launch(
            set_kinematic_particles,
            dim=self.controller_particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.controller_start_index,
                self.current_controller_positions,
                self.current_controller_velocities,
            ],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_controller_targets(self.state_0)
            self.state_0.clear_forces()

            if self.drag_damping > 0.0:
                wp.launch(
                    apply_particle_drag,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_0.particle_qd,
                        self.model.particle_mass,
                        self.drag_damping,
                        self.sim_dt,
                    ],
                    device=self.model.device,
                )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._set_controller_frame(self.current_frame_index)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.current_frame_index += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.show_springs:
            particle_q = self.state_0.particle_q.numpy()
            self.viewer.log_lines(
                "/model/springs",
                wp.array(particle_q[self.spring_indices[:, 0]], dtype=wp.vec3),
                wp.array(particle_q[self.spring_indices[:, 1]], dtype=wp.vec3),
                colors=(0.80, 0.83, 0.92),
                width=max(0.002, 0.15 * self.particle_radius),
            )
        else:
            self.viewer.log_lines("/model/springs", None, None, None)

        if self.controller_particle_count:
            particle_q = self.state_0.particle_q.numpy()
            controller_q = particle_q[self.controller_start_index :].astype(np.float32)

            controller_radii = np.full(
                self.controller_particle_count,
                max(0.006, 1.5 * self.particle_radius),
                dtype=np.float32,
            )

            controller_colors = np.tile(
                np.array([[0.90, 0.25, 0.20]], dtype=np.float32),
                (self.controller_particle_count, 1),
            )

            self.viewer.log_points(
                "/model/controller_points",
                wp.array(controller_q, dtype=wp.vec3),
                radii=wp.array(controller_radii, dtype=wp.float32),
                colors=wp.array(controller_colors, dtype=wp.vec3),
            )
        else:
            self.viewer.log_points("/model/controller_points", None, None, None)

        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        object_q = particle_q[: self.object_particle_count]
        object_qd = particle_qd[: self.object_particle_count]

        assert np.isfinite(particle_q).all()
        assert np.isfinite(particle_qd).all()

        max_speed = np.max(np.linalg.norm(object_qd, axis=1))
        assert max_speed < 100.0, f"Object particle speeds exploded: max_speed={max_speed:.4f}"

        bbox_size = np.linalg.norm(np.max(object_q, axis=0) - np.min(object_q, axis=0))
        assert bbox_size < 10.0, f"Object particle bounding box exploded: size={bbox_size:.4f}"

        min_height = float(np.min(object_q[:, 2]))
        assert min_height > -1.0, f"Object particles penetrated too far below ground: z_min={min_height:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--npz-path",
            type=str,
            default=str(DEFAULT_NPZ_PATH),
            help="Path to the PhysTwin export `.npz` file.",
        )
        parser.add_argument(
            "--stiffness-mode",
            type=str,
            choices=["auto", "exp", "direct"],
            default="auto",
            help="How to interpret `spring_Y`: auto-detect, exp(log-stiffness), or direct stiffness.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=None,
            help="Override the number of physics substeps per controller frame while preserving controller timing.",
        )
        parser.add_argument(
            "--show-springs",
            action="store_true",
            default=True,
            help="Render spring segments as debug lines.",
        )
        parser.add_argument(
            "--no-show-springs",
            action="store_false",
            dest="show_springs",
            help="Disable spring-line rendering.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
