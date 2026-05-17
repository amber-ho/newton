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

"""Loader for PhysTwin physics export data into Newton simulation structures.

This module provides utilities to load PhysTwin-exported physics parameters
(stored in .npz format) and construct Newton-compatible simulation states.

Example:
    >>> from newton import load_phystwin_npz
    >>> # Load physics data from PhysTwin export
    >>> physics_data = load_phystwin_npz.load_physics_data("physics_params.npz")
    >>> # Create Newton model and state
    >>> model = newton.ModelBuilder().build()
    >>> state = load_phystwin_npz.create_state_from_physics_data(model, physics_data)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PhysicsData:
    """Container for PhysTwin physics export data.
    
    Attributes:
        object_vertices_0: Initial particle positions, shape (N, 3)
        controller_trajectory: Controller kinematic points trajectory, shape (T, M, 3)
        springs: Spring indices, shape (S, 2), dtype int32
        rest_lengths: Spring rest lengths [m], shape (S,)
        spring_Y: Spring stiffness parameters (log Young's modulus), shape (S,)
        masses: Particle masses [kg], shape (N,)
        
        Optional parameters:
        collide_elas: Collision elasticity coefficient [dimensionless]
        collide_fric: Collision friction coefficient [dimensionless]
        collision_dist: Collision detection distance [m]
        drag_damping: Drag damping coefficient [1/s]
        dashpot_damping: Dashpot damping coefficient [1/s]
        dt: Simulation timestep [s]
        num_substeps: Number of substeps per frame
    """
    object_vertices_0: np.ndarray
    controller_trajectory: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    spring_Y: np.ndarray
    masses: np.ndarray
    
    # Optional parameters
    collide_elas: float = 0.3
    collide_fric: float = 0.3
    collision_dist: float = 0.02
    drag_damping: float = 0.0
    dashpot_damping: float = 0.0
    dt: float = 1.0 / 30.0
    num_substeps: int = 1


def load_physics_data(npz_path: str) -> PhysicsData:
    """Load PhysTwin physics export data from .npz file.
    
    Args:
        npz_path: Path to the .npz file containing physics parameters.
        
    Returns:
        PhysicsData: Container with all loaded physics parameters.
        
    Raises:
        FileNotFoundError: If the .npz file does not exist.
        KeyError: If required fields are missing from the .npz file.
    """
    data = np.load(npz_path)
    
    # Extract required fields
    required_fields = {
        "object_vertices_0": "Initial particle positions",
        "controller_trajectory": "Controller trajectory",
        "springs": "Spring indices",
        "rest_lengths": "Spring rest lengths",
        "spring_Y": "Spring stiffness parameters",
        "masses": "Particle masses",
    }
    
    for field, description in required_fields.items():
        if field not in data:
            raise KeyError(f"Required field '{field}' ({description}) not found in {npz_path}")
    
    # Extract optional fields with defaults
    optional_fields = {
        "collide_elas": 0.3,
        "collide_fric": 0.3,
        "collision_dist": 0.02,
        "drag_damping": 0.0,
        "dashpot_damping": 0.0,
        "dt": 1.0 / 30.0,
        "num_substeps": 1,
    }
    
    kwargs = {}
    for field, default in optional_fields.items():
        if field in data:
            kwargs[field] = float(data[field])
        else:
            kwargs[field] = default
    
    physics_data = PhysicsData(
        object_vertices_0=np.array(data["object_vertices_0"], dtype=np.float32),
        controller_trajectory=np.array(data["controller_trajectory"], dtype=np.float32),
        springs=np.array(data["springs"], dtype=np.int32),
        rest_lengths=np.array(data["rest_lengths"], dtype=np.float32),
        spring_Y=np.array(data["spring_Y"], dtype=np.float32),
        masses=np.array(data["masses"], dtype=np.float32),
        **kwargs
    )
    
    return physics_data


def create_particle_states(physics_data: PhysicsData) -> Dict[str, np.ndarray]:
    """Create Newton-compatible particle state from PhysTwin physics data.
    
    Constructs particle positions, velocities, and masses from the loaded
    physics data. Initial velocities are set to zero.
    
    Args:
        physics_data: PhysicsData container with loaded physics parameters.
        
    Returns:
        Dictionary with keys:
            - "positions": Particle positions, shape (N, 3), dtype float32
            - "velocities": Particle velocities (initialized to zero), shape (N, 3), dtype float32
            - "masses": Particle masses [kg], shape (N,), dtype float32
    """
    num_particles = physics_data.object_vertices_0.shape[0]
    
    # Validate masses
    if physics_data.masses.shape[0] != num_particles:
        raise ValueError(
            f"Mismatch: {num_particles} particles but {physics_data.masses.shape[0]} masses"
        )
    
    particle_states = {
        "positions": physics_data.object_vertices_0.copy(),
        "velocities": np.zeros((num_particles, 3), dtype=np.float32),
        "masses": physics_data.masses.copy(),
    }
    
    return particle_states


def create_spring_topology(physics_data: PhysicsData) -> Dict[str, np.ndarray]:
    """Create Newton-compatible spring topology from PhysTwin physics data.
    
    Converts spring definitions to Newton format with stiffness and damping
    properties derived from the Young's modulus parameter (spring_Y).
    
    Args:
        physics_data: PhysicsData container with loaded physics parameters.
        
    Returns:
        Dictionary with keys:
            - "indices": Spring endpoint indices, shape (S, 2), dtype int32
            - "rest_lengths": Spring rest lengths [m], shape (S,), dtype float32
            - "stiffness": Spring stiffness [N/m], shape (S,), dtype float32
            - "damping": Spring damping [N·s/m], shape (S,), dtype float32
            
    Notes:
        - Stiffness is derived from spring_Y: stiffness = exp(spring_Y)
        - Damping is set to 0.1 * stiffness (can be customized)
        - Spring indices should be in range [0, num_particles-1]
    """
    num_springs = physics_data.springs.shape[0]
    
    # Validate spring_Y
    if physics_data.spring_Y.shape[0] != num_springs:
        raise ValueError(
            f"Mismatch: {num_springs} springs but {physics_data.spring_Y.shape[0]} spring_Y values"
        )
    
    # Validate rest_lengths
    if physics_data.rest_lengths.shape[0] != num_springs:
        raise ValueError(
            f"Mismatch: {num_springs} springs but {physics_data.rest_lengths.shape[0]} rest_lengths"
        )
    
    # Convert spring_Y (log Young's modulus) to stiffness
    # spring_Y typically ranges from ~7-12, corresponding to ~1000-150000 N/m
    stiffness = np.exp(physics_data.spring_Y).astype(np.float32)
    
    # Set damping as a fraction of stiffness (critical damping factor)
    # Can be adjusted: higher values = more damping
    damping = (0.1 * stiffness).astype(np.float32)
    
    spring_topology = {
        "indices": physics_data.springs.copy(),
        "rest_lengths": physics_data.rest_lengths.copy(),
        "stiffness": stiffness,
        "damping": damping,
    }
    
    return spring_topology


def create_controller_trajectory(physics_data: PhysicsData) -> Optional[np.ndarray]:
    """Extract controller kinematic points trajectory from PhysTwin data.
    
    Returns the controller trajectory for keyframe-based control or reference.
    
    Args:
        physics_data: PhysicsData container with loaded physics parameters.
        
    Returns:
        Controller trajectory array with shape (T, M, 3), dtype float32,
        where T is number of timesteps and M is number of control points.
        Returns empty array if controller_trajectory is not available.
    """
    if physics_data.controller_trajectory.size == 0:
        return np.array([], dtype=np.float32)
    
    return physics_data.controller_trajectory.copy()


def get_simulation_parameters(physics_data: PhysicsData) -> Dict[str, Any]:
    """Extract simulation parameters from PhysTwin physics data.
    
    Args:
        physics_data: PhysicsData container with loaded physics parameters.
        
    Returns:
        Dictionary with simulation parameters:
            - "dt": Timestep [s]
            - "num_substeps": Number of substeps per frame
            - "collision_dist": Collision detection distance [m]
            - "drag_damping": Drag damping coefficient [1/s]
            - "dashpot_damping": Dashpot damping coefficient [1/s]
            - "collide_elas": Collision elasticity [dimensionless]
            - "collide_fric": Collision friction [dimensionless]
    """
    return {
        "dt": physics_data.dt,
        "num_substeps": physics_data.num_substeps,
        "collision_dist": physics_data.collision_dist,
        "drag_damping": physics_data.drag_damping,
        "dashpot_damping": physics_data.dashpot_damping,
        "collide_elas": physics_data.collide_elas,
        "collide_fric": physics_data.collide_fric,
    }


def load_and_create_states(npz_path: str) -> Dict[str, Any]:
    """Convenience function to load PhysTwin data and create all Newton states.
    
    This is the main entry point for loading PhysTwin physics exports.
    It loads the .npz file and creates all necessary Newton-compatible
    structures in a single call.
    
    Args:
        npz_path: Path to the .npz file containing physics parameters.
        
    Returns:
        Dictionary containing:
            - "physics_data": PhysicsData object with raw loaded data
            - "particles": Particle states (positions, velocities, masses)
            - "springs": Spring topology (indices, rest_lengths, stiffness, damping)
            - "controller_trajectory": Controller kinematic trajectory
            - "simulation_params": Simulation parameters
            
    Example:
        >>> states = load_and_create_states("physics_params.npz")
        >>> particle_pos = states["particles"]["positions"]
        >>> spring_indices = states["springs"]["indices"]
        >>> sim_params = states["simulation_params"]
    """
    # Load raw data from .npz
    physics_data = load_physics_data(npz_path)
    
    # Create Newton-compatible structures
    particles = create_particle_states(physics_data)
    springs = create_spring_topology(physics_data)
    controller_trajectory = create_controller_trajectory(physics_data)
    sim_params = get_simulation_parameters(physics_data)
    
    return {
        "physics_data": physics_data,
        "particles": particles,
        "springs": springs,
        "controller_trajectory": controller_trajectory,
        "simulation_params": sim_params,
    }
