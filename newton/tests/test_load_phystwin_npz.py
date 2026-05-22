"""Unit tests for load_phystwin_npz module.

Tests the loading and state creation from PhysTwin physics exports.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from newton import load_phystwin_npz


class TestLoadPhysicsData:
    """Tests for load_physics_data function."""
    
    def create_test_npz(self, tmp_path: Path) -> str:
        """Create a test .npz file with valid physics data."""
        num_particles = 100
        num_springs = 50
        num_timesteps = 10
        num_control_points = 4
        
        data = {
            "object_vertices_0": np.random.randn(num_particles, 3).astype(np.float32),
            "controller_trajectory": np.random.randn(
                num_timesteps, num_control_points, 3
            ).astype(np.float32),
            "springs": np.random.randint(0, num_particles, (num_springs, 2)).astype(np.int32),
            "rest_lengths": np.random.uniform(0.01, 0.5, num_springs).astype(np.float32),
            "spring_Y": np.random.uniform(7, 12, num_springs).astype(np.float32),
            "masses": np.random.uniform(0.001, 0.1, num_particles).astype(np.float32),
            # Optional parameters
            "collide_elas": np.float32(0.3),
            "collide_fric": np.float32(0.3),
            "collision_dist": np.float32(0.02),
            "drag_damping": np.float32(0.01),
            "dashpot_damping": np.float32(5.0),
            "dt": np.float32(1.0 / 30.0),
            "num_substeps": np.int32(1),
        }
        
        npz_path = str(tmp_path / "test_physics.npz")
        np.savez(npz_path, **data)
        return npz_path
    
    def test_load_physics_data_success(self, tmp_path: Path):
        """Test successful loading of valid physics data."""
        npz_path = self.create_test_npz(tmp_path)
        
        physics_data = load_phystwin_npz.load_physics_data(npz_path)
        
        assert physics_data.object_vertices_0.shape[0] == 100
        assert physics_data.springs.shape[0] == 50
        assert physics_data.masses.shape[0] == 100
        assert physics_data.collide_elas == 0.3
        assert physics_data.dt == pytest.approx(1.0 / 30.0)
    
    def test_load_physics_data_missing_required_field(self, tmp_path: Path):
        """Test that KeyError is raised when required field is missing."""
        # Create incomplete data
        data = {
            "object_vertices_0": np.random.randn(10, 3),
            "springs": np.random.randint(0, 10, (5, 2)),
            # Missing rest_lengths, spring_Y, masses, controller_trajectory
        }
        npz_path = str(tmp_path / "incomplete.npz")
        np.savez(npz_path, **data)
        
        with pytest.raises(KeyError):
            load_phystwin_npz.load_physics_data(npz_path)
    
    def test_load_physics_data_defaults(self, tmp_path: Path):
        """Test that optional parameters use defaults when missing."""
        num_particles = 10
        num_springs = 5
        
        data = {
            "object_vertices_0": np.random.randn(num_particles, 3).astype(np.float32),
            "controller_trajectory": np.empty((0, 0, 3), dtype=np.float32),
            "springs": np.random.randint(0, num_particles, (num_springs, 2)).astype(np.int32),
            "rest_lengths": np.ones(num_springs, dtype=np.float32),
            "spring_Y": np.ones(num_springs, dtype=np.float32) * 9,
            "masses": np.ones(num_particles, dtype=np.float32) * 0.01,
        }
        
        npz_path = str(tmp_path / "minimal.npz")
        np.savez(npz_path, **data)
        
        physics_data = load_phystwin_npz.load_physics_data(npz_path)
        
        assert physics_data.collide_elas == 0.3  # default
        assert physics_data.dt == pytest.approx(1.0 / 30.0)  # default
        assert physics_data.num_substeps == 1  # default


class TestCreateParticleStates:
    """Tests for create_particle_states function."""
    
    def test_create_particle_states(self):
        """Test particle state creation."""
        num_particles = 50
        
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(num_particles, 3).astype(np.float32),
            controller_trajectory=np.empty((0, 0, 3), dtype=np.float32),
            springs=np.array([[0, 1]], dtype=np.int32),
            rest_lengths=np.array([0.1], dtype=np.float32),
            spring_Y=np.array([9.0], dtype=np.float32),
            masses=np.random.uniform(0.001, 0.1, num_particles).astype(np.float32),
        )
        
        states = load_phystwin_npz.create_particle_states(physics_data)
        
        assert states["positions"].shape == (num_particles, 3)
        assert states["velocities"].shape == (num_particles, 3)
        assert states["masses"].shape == (num_particles,)
        
        # Check initial velocities are zero
        assert np.allclose(states["velocities"], 0.0)
        
        # Check dtypes
        assert states["positions"].dtype == np.float32
        assert states["velocities"].dtype == np.float32
        assert states["masses"].dtype == np.float32
    
    def test_particle_states_mass_mismatch(self):
        """Test error when masses don't match particle count."""
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(50, 3).astype(np.float32),
            controller_trajectory=np.empty((0, 0, 3), dtype=np.float32),
            springs=np.array([[0, 1]], dtype=np.int32),
            rest_lengths=np.array([0.1], dtype=np.float32),
            spring_Y=np.array([9.0], dtype=np.float32),
            masses=np.ones(30, dtype=np.float32),  # Wrong count!
        )
        
        with pytest.raises(ValueError, match="Mismatch"):
            load_phystwin_npz.create_particle_states(physics_data)


class TestCreateSpringTopology:
    """Tests for create_spring_topology function."""
    
    def test_create_spring_topology(self):
        """Test spring topology creation."""
        num_particles = 100
        num_springs = 50
        
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(num_particles, 3).astype(np.float32),
            controller_trajectory=np.empty((0, 0, 3), dtype=np.float32),
            springs=np.random.randint(0, num_particles, (num_springs, 2)).astype(np.int32),
            rest_lengths=np.random.uniform(0.01, 0.5, num_springs).astype(np.float32),
            spring_Y=np.random.uniform(7, 12, num_springs).astype(np.float32),
            masses=np.ones(num_particles, dtype=np.float32) * 0.01,
        )
        
        topology = load_phystwin_npz.create_spring_topology(physics_data)
        
        assert topology["indices"].shape == (num_springs, 2)
        assert topology["rest_lengths"].shape == (num_springs,)
        assert topology["stiffness"].shape == (num_springs,)
        assert topology["damping"].shape == (num_springs,)
        
        # Check that stiffness = exp(spring_Y)
        expected_stiffness = np.exp(physics_data.spring_Y)
        assert np.allclose(topology["stiffness"], expected_stiffness, rtol=1e-5)
        
        # Check that damping = 0.1 * stiffness
        expected_damping = 0.1 * expected_stiffness
        assert np.allclose(topology["damping"], expected_damping, rtol=1e-5)
        
        # Check all stiffness values are positive
        assert np.all(topology["stiffness"] > 0)
        assert np.all(topology["damping"] > 0)
    
    def test_spring_topology_spring_count_mismatch(self):
        """Test error when spring counts don't match."""
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(100, 3).astype(np.float32),
            controller_trajectory=np.empty((0, 0, 3), dtype=np.float32),
            springs=np.array([[0, 1], [2, 3]], dtype=np.int32),  # 2 springs
            rest_lengths=np.array([0.1, 0.2, 0.3], dtype=np.float32),  # 3 rest lengths!
            spring_Y=np.array([9.0, 9.5], dtype=np.float32),
            masses=np.ones(100, dtype=np.float32) * 0.01,
        )
        
        with pytest.raises(ValueError, match="Mismatch"):
            load_phystwin_npz.create_spring_topology(physics_data)


class TestCreateControllerTrajectory:
    """Tests for create_controller_trajectory function."""
    
    def test_create_controller_trajectory_with_data(self):
        """Test controller trajectory extraction when available."""
        num_timesteps = 10
        num_control_points = 4
        
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(50, 3).astype(np.float32),
            controller_trajectory=np.random.randn(
                num_timesteps, num_control_points, 3
            ).astype(np.float32),
            springs=np.array([[0, 1]], dtype=np.int32),
            rest_lengths=np.array([0.1], dtype=np.float32),
            spring_Y=np.array([9.0], dtype=np.float32),
            masses=np.ones(50, dtype=np.float32),
        )
        
        trajectory = load_phystwin_npz.create_controller_trajectory(physics_data)
        
        assert trajectory.shape == (num_timesteps, num_control_points, 3)
        assert trajectory.dtype == np.float32
    
    def test_create_controller_trajectory_empty(self):
        """Test empty controller trajectory handling."""
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(50, 3).astype(np.float32),
            controller_trajectory=np.array([], dtype=np.float32),
            springs=np.array([[0, 1]], dtype=np.int32),
            rest_lengths=np.array([0.1], dtype=np.float32),
            spring_Y=np.array([9.0], dtype=np.float32),
            masses=np.ones(50, dtype=np.float32),
        )
        
        trajectory = load_phystwin_npz.create_controller_trajectory(physics_data)
        
        assert trajectory.size == 0
        assert trajectory.dtype == np.float32


class TestGetSimulationParameters:
    """Tests for get_simulation_parameters function."""
    
    def test_get_simulation_parameters(self):
        """Test simulation parameter extraction."""
        physics_data = load_phystwin_npz.PhysicsData(
            object_vertices_0=np.random.randn(50, 3).astype(np.float32),
            controller_trajectory=np.empty((0, 0, 3), dtype=np.float32),
            springs=np.array([[0, 1]], dtype=np.int32),
            rest_lengths=np.array([0.1], dtype=np.float32),
            spring_Y=np.array([9.0], dtype=np.float32),
            masses=np.ones(50, dtype=np.float32),
            dt=0.01,
            num_substeps=3,
            collision_dist=0.05,
            drag_damping=0.02,
            dashpot_damping=10.0,
            collide_elas=0.5,
            collide_fric=0.4,
        )
        
        params = load_phystwin_npz.get_simulation_parameters(physics_data)
        
        assert params["dt"] == pytest.approx(0.01)
        assert params["num_substeps"] == 3
        assert params["collision_dist"] == pytest.approx(0.05)
        assert params["drag_damping"] == pytest.approx(0.02)
        assert params["dashpot_damping"] == pytest.approx(10.0)
        assert params["collide_elas"] == pytest.approx(0.5)
        assert params["collide_fric"] == pytest.approx(0.4)


class TestLoadAndCreateStates:
    """Tests for load_and_create_states integration function."""
    
    def test_load_and_create_states(self, tmp_path: Path):
        """Test full workflow of loading and creating states."""
        num_particles = 100
        num_springs = 50
        num_timesteps = 10
        num_control_points = 4
        
        data = {
            "object_vertices_0": np.random.randn(num_particles, 3).astype(np.float32),
            "controller_trajectory": np.random.randn(
                num_timesteps, num_control_points, 3
            ).astype(np.float32),
            "springs": np.random.randint(0, num_particles, (num_springs, 2)).astype(np.int32),
            "rest_lengths": np.random.uniform(0.01, 0.5, num_springs).astype(np.float32),
            "spring_Y": np.random.uniform(7, 12, num_springs).astype(np.float32),
            "masses": np.random.uniform(0.001, 0.1, num_particles).astype(np.float32),
            "dt": np.float32(1.0 / 30.0),
            "num_substeps": np.int32(1),
        }
        
        npz_path = str(tmp_path / "test_physics.npz")
        np.savez(npz_path, **data)
        
        result = load_phystwin_npz.load_and_create_states(npz_path)
        
        # Check structure
        assert "physics_data" in result
        assert "particles" in result
        assert "springs" in result
        assert "controller_trajectory" in result
        assert "simulation_params" in result
        
        # Check particle data
        assert result["particles"]["positions"].shape == (num_particles, 3)
        assert result["particles"]["velocities"].shape == (num_particles, 3)
        assert result["particles"]["masses"].shape == (num_particles,)
        
        # Check spring data
        assert result["springs"]["indices"].shape == (num_springs, 2)
        assert result["springs"]["rest_lengths"].shape == (num_springs,)
        assert result["springs"]["stiffness"].shape == (num_springs,)
        assert result["springs"]["damping"].shape == (num_springs,)
        
        # Check controller trajectory
        assert result["controller_trajectory"].shape == (num_timesteps, num_control_points, 3)
        
        # Check simulation parameters
        assert "dt" in result["simulation_params"]
        assert "num_substeps" in result["simulation_params"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
