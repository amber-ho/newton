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
# Example Cloth Bridge
#
# This simulation demonstrates a compact workflow for building a cloth scene:
# create a model builder, add static collision geometry, add a cloth grid,
# choose a solver, allocate states, and step the simulation.
#
# The cloth is suspended between two supports and drapes over a spherical
# obstacle, making it a useful starting point for custom cloth scenes.
#
# Command: uv run -m newton.examples cloth_bridge
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 6

        self.dim_x = 16
        self.dim_y = 8
        self.anchor_height = 1.3
        self.span = 2.4
        self.cloth_width = 1.1
        self.cell_x = self.span / self.dim_x
        self.cell_y = self.cloth_width / self.dim_y
        self.center_particle_index = (self.dim_y // 2) * (self.dim_x + 1) + self.dim_x // 2

        builder = newton.ModelBuilder()

        support_cfg = newton.ModelBuilder.ShapeConfig()
        support_cfg.density = 0.0
        support_cfg.ke = 5.0e4
        support_cfg.kd = 5.0e1
        support_cfg.mu = 0.8

        self._add_support(
            builder,
            pos=wp.vec3(-0.16, 0.0, 0.85),
            hx=0.12,
            hy=0.45,
            hz=0.85,
            cfg=support_cfg,
            label="support_left",
        )
        self._add_support(
            builder,
            pos=wp.vec3(self.span + 0.16, 0.0, 0.85),
            hx=0.12,
            hy=0.45,
            hz=0.85,
            cfg=support_cfg,
            label="support_right",
        )

        obstacle_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(self.span * 0.5, 0.0, 0.55), q=wp.quat_identity()),
            label="obstacle",
        )
        builder.add_shape_sphere(
            obstacle_body,
            radius=0.35,
            cfg=support_cfg,
        )
        self.obstacle_top = 0.55 + 0.35

        builder.add_ground_plane()

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, -0.5 * self.cloth_width, self.anchor_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            mass=0.03,
            fix_left=True,
            fix_right=True,
            tri_ke=4.0e3,
            tri_ka=4.0e3,
            tri_kd=5.0e-2,
            edge_ke=3.0e1,
            edge_kd=1.0,
            particle_radius=0.035,
        )

        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 5.0e1
        self.model.soft_contact_mu = 0.8

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.02,
            particle_self_contact_margin=0.03,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(4.2, -2.8, 2.0),
            pitch=-18.0,
            yaw=145.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

        self.capture()

    @staticmethod
    def _add_support(builder, pos, hx, hy, hz, cfg, label):
        body = builder.add_body(
            xform=wp.transform(p=pos, q=wp.quat_identity()),
            label=label,
        )
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()

        left_edge = positions[0 :: self.dim_x + 1]
        right_edge = positions[self.dim_x :: self.dim_x + 1]
        center_particle = positions[self.center_particle_index]

        assert np.max(np.abs(left_edge[:, 2] - self.anchor_height)) < 1.0e-5
        assert np.max(np.abs(right_edge[:, 2] - self.anchor_height)) < 1.0e-5
        assert center_particle[2] < self.anchor_height - 0.15
        assert center_particle[2] > self.obstacle_top - 0.05
        assert positions[:, 2].min() > -0.01
        assert np.max(np.linalg.norm(velocities, axis=1)) < 3.0

        x_min = positions[:, 0].min()
        x_max = positions[:, 0].max()
        assert x_min > -0.05
        assert x_max < self.span + 0.05


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
