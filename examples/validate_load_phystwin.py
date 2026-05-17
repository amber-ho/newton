"""Simple validation script to verify load_phystwin_npz works correctly.

This creates a test .npz file and validates the loader functions.
"""

import numpy as np
import tempfile
from pathlib import Path

from newton import load_phystwin_npz


def main():
    """Create test data and validate all loader functions."""
    
    print("=" * 70)
    print("Newton PhysTwin NPZ Loader Validation")
    print("=" * 70)
    
    # Create test data
    print("\n1. Creating test physics data...")
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
        "collide_elas": np.float32(0.3),
        "collide_fric": np.float32(0.3),
        "collision_dist": np.float32(0.02),
        "drag_damping": np.float32(0.01),
        "dashpot_damping": np.float32(5.0),
        "dt": np.float32(1.0 / 30.0),
        "num_substeps": np.int32(1),
    }
    
    # Create temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = str(Path(tmpdir) / "test_physics.npz")
        np.savez(npz_path, **data)
        print(f"   ✓ Created test file: {npz_path}")
        
        # Test 1: Load physics data
        print("\n2. Testing load_physics_data()...")
        try:
            physics_data = load_phystwin_npz.load_physics_data(npz_path)
            print(f"   ✓ Loaded physics data successfully")
            print(f"     - Particles: {physics_data.object_vertices_0.shape[0]}")
            print(f"     - Springs: {physics_data.springs.shape[0]}")
            print(f"     - dt: {physics_data.dt}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test 2: Create particle states
        print("\n3. Testing create_particle_states()...")
        try:
            particles = load_phystwin_npz.create_particle_states(physics_data)
            assert particles["positions"].shape == (num_particles, 3)
            assert particles["velocities"].shape == (num_particles, 3)
            assert particles["masses"].shape == (num_particles,)
            assert np.allclose(particles["velocities"], 0.0)
            print(f"   ✓ Created particle states successfully")
            print(f"     - Positions: {particles['positions'].shape}")
            print(f"     - Velocities: {particles['velocities'].shape}")
            print(f"     - Masses: {particles['masses'].shape}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test 3: Create spring topology
        print("\n4. Testing create_spring_topology()...")
        try:
            springs = load_phystwin_npz.create_spring_topology(physics_data)
            assert springs["indices"].shape == (num_springs, 2)
            assert springs["rest_lengths"].shape == (num_springs,)
            assert springs["stiffness"].shape == (num_springs,)
            assert springs["damping"].shape == (num_springs,)
            
            # Verify stiffness = exp(spring_Y)
            expected_stiffness = np.exp(physics_data.spring_Y)
            assert np.allclose(springs["stiffness"], expected_stiffness, rtol=1e-5)
            assert np.all(springs["stiffness"] > 0)
            assert np.all(springs["damping"] > 0)
            
            print(f"   ✓ Created spring topology successfully")
            print(f"     - Spring indices: {springs['indices'].shape}")
            print(f"     - Stiffness range: [{springs['stiffness'].min():.1f}, {springs['stiffness'].max():.1f}] N/m")
            print(f"     - Damping range: [{springs['damping'].min():.1f}, {springs['damping'].max():.1f}] N·s/m")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test 4: Create controller trajectory
        print("\n5. Testing create_controller_trajectory()...")
        try:
            trajectory = load_phystwin_npz.create_controller_trajectory(physics_data)
            assert trajectory.shape == (num_timesteps, num_control_points, 3)
            assert trajectory.dtype == np.float32
            print(f"   ✓ Created controller trajectory successfully")
            print(f"     - Shape: {trajectory.shape}")
            print(f"     - Timesteps: {trajectory.shape[0]}, Control points: {trajectory.shape[1]}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test 5: Get simulation parameters
        print("\n6. Testing get_simulation_parameters()...")
        try:
            params = load_phystwin_npz.get_simulation_parameters(physics_data)
            required_keys = [
                "dt", "num_substeps", "collision_dist",
                "drag_damping", "dashpot_damping", "collide_elas", "collide_fric"
            ]
            for key in required_keys:
                assert key in params, f"Missing parameter: {key}"
            print(f"   ✓ Retrieved simulation parameters successfully")
            print(f"     - dt: {params['dt']:.6f} s")
            print(f"     - num_substeps: {params['num_substeps']}")
            print(f"     - collision_dist: {params['collision_dist']:.6f} m")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test 6: Full workflow
        print("\n7. Testing load_and_create_states() (full workflow)...")
        try:
            result = load_phystwin_npz.load_and_create_states(npz_path)
            
            assert "physics_data" in result
            assert "particles" in result
            assert "springs" in result
            assert "controller_trajectory" in result
            assert "simulation_params" in result
            
            # Verify all structures
            assert result["particles"]["positions"].shape == (num_particles, 3)
            assert result["springs"]["indices"].shape == (num_springs, 2)
            assert result["controller_trajectory"].shape == (num_timesteps, num_control_points, 3)
            
            print(f"   ✓ Full workflow completed successfully")
            print(f"     - Physics data loaded")
            print(f"     - Particle states created: {num_particles} particles")
            print(f"     - Spring topology created: {num_springs} springs")
            print(f"     - Controller trajectory extracted")
            print(f"     - Simulation parameters retrieved")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Validation checks
        print("\n8. Data validation checks...")
        try:
            particles = result["particles"]
            springs = result["springs"]
            
            # Check spring indices validity
            max_particle_idx = particles["positions"].shape[0] - 1
            invalid_springs = np.where(
                (springs["indices"][:, 0] > max_particle_idx) |
                (springs["indices"][:, 1] > max_particle_idx) |
                (springs["indices"][:, 0] < 0) |
                (springs["indices"][:, 1] < 0)
            )[0]
            
            if len(invalid_springs) == 0:
                print(f"   ✓ All spring indices are valid")
            else:
                print(f"   ⚠ Warning: {len(invalid_springs)} springs with invalid indices")
            
            # Check for zero masses
            zero_mass_particles = np.where(particles["masses"] == 0)[0]
            if len(zero_mass_particles) == 0:
                print(f"   ✓ All particles have non-zero mass")
            else:
                print(f"   ⚠ Warning: {len(zero_mass_particles)} particles with zero mass")
            
            # Check stiffness values
            if np.all(springs["stiffness"] > 0):
                print(f"   ✓ All spring stiffness values are positive")
            else:
                print(f"   ✗ Error: Found non-positive stiffness values")
                return False
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    print("\n" + "=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
