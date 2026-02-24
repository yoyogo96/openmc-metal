"""Persistent kernel k-eigenvalue simulation driver."""

import time
import math
import numpy as np

from .engine import PersistentEngine
from ..cross_sections import c5g7_all_materials
from ..geometry import (
    build_c5g7_pincell, build_c5g7_assembly,
    source_sampler, assembly_source_sampler,
    find_cell_pincell, find_cell_assembly,
    REFERENCE_KEFF, ASSEMBLY_REFERENCE_KEFF,
    Surface, Cell,
)
from ..statistics import keff_statistics


class PersistentSimulation:
    """K-eigenvalue simulation using history-based persistent GPU kernel."""

    def __init__(self, num_particles=100_000, num_batches=100,
                 num_inactive=50, num_groups=7):
        self.num_particles = num_particles
        self.num_batches = num_batches
        self.num_inactive = num_inactive
        self.num_groups = num_groups

        self.engine = PersistentEngine()
        self._prepare_materials()

    def _prepare_materials(self):
        """Flatten C5G7 materials into separate SoA arrays for GPU."""
        materials = c5g7_all_materials()
        n_mats = len(materials)  # 7
        ng = self.num_groups     # 7

        self.sigma_t = np.zeros(n_mats * ng, dtype=np.float32)
        self.sigma_s = np.zeros(n_mats * ng * ng, dtype=np.float32)
        self.sigma_a = np.zeros(n_mats * ng, dtype=np.float32)
        self.sigma_f = np.zeros(n_mats * ng, dtype=np.float32)
        self.nu_sf = np.zeros(n_mats * ng, dtype=np.float32)
        self.chi = np.zeros(n_mats * ng, dtype=np.float32)
        self.is_fiss = np.zeros(n_mats, dtype=np.float32)

        for m, mat in enumerate(materials):
            for g in range(ng):
                self.sigma_t[m * ng + g] = mat['total'][g]
                self.sigma_f[m * ng + g] = mat['fission'][g]
                self.nu_sf[m * ng + g] = mat['nu_fission'][g]
                self.chi[m * ng + g] = mat['chi'][g]

                # scatter matrix: mat['scatter'] is flat list of 49 values
                # Layout: row-major, scatter[from_g * 7 + to_g]
                for g2 in range(ng):
                    self.sigma_s[m * ng * ng + g * ng + g2] = mat['scatter'][g * ng + g2]

                # absorption = total - sum of scatter row
                scatter_sum = sum(mat['scatter'][g * ng + g2] for g2 in range(ng))
                self.sigma_a[m * ng + g] = mat['total'][g] - scatter_sum

            # Is fissile if any fission XS > 0
            self.is_fiss[m] = 1.0 if any(mat['fission'][g] > 0 for g in range(ng)) else 0.0

        # Create GPU buffers for materials once (reused every batch)
        self._sigma_t_buf = self.engine.make_buffer_from_numpy(self.sigma_t)
        self._sigma_s_buf = self.engine.make_buffer_from_numpy(self.sigma_s)
        self._sigma_a_buf = self.engine.make_buffer_from_numpy(self.sigma_a)
        self._sigma_f_buf = self.engine.make_buffer_from_numpy(self.sigma_f)
        self._nu_sf_buf = self.engine.make_buffer_from_numpy(self.nu_sf)
        self._chi_buf = self.engine.make_buffer_from_numpy(self.chi)
        self._is_fiss_buf = self.engine.make_buffer_from_numpy(self.is_fiss)

    def _prepare_geometry(self, geometry_type):
        """Pack geometry into flat numpy arrays for the persistent kernel.

        The persistent shader reads geometry as flat arrays:
        - surfaces: 8 floats per surface [type, bc, cx, cy, cz, cw, pad, pad]
        - cells: 4 uints per cell [material_index, num_surfaces, surface_offset, pad]
        - cell_surfs: 2 ints per entry [surface_index, sense]
        """
        if geometry_type == 'pincell':
            surfaces, cells = self._build_pincell_raw()
            self.cell_finder = find_cell_pincell
            self.source_sampler_fn = source_sampler
            self.reference_keff = REFERENCE_KEFF
        else:
            surfaces, cells = self._build_assembly_raw()
            self.cell_finder = find_cell_assembly
            self.source_sampler_fn = assembly_source_sampler
            self.reference_keff = ASSEMBLY_REFERENCE_KEFF

        # Pack surfaces: 8 floats per surface
        # Shader layout: [type, bc, cx, cy, cz, cw, pad, pad]
        # For CYLINDER_Z: cx=x0, cy=y0, cz=unused, cw=radius
        # For SPHERE:     cx=x0, cy=y0, cz=z0,     cw=radius
        # But Surface.cylinder_z stores coefficients=(x0, y0, radius, 0.0)
        # so we must remap: coeff[2] (radius) → index 5 (cw)
        from ..types import SurfaceType as ST

        n_surfs = len(surfaces)
        surf_flat = np.zeros(n_surfs * 8, dtype=np.float32)
        for i, s in enumerate(surfaces):
            base = i * 8
            # Store type and bc as float (shader reads as uint via as_type)
            surf_flat[base + 0] = np.float32(np.frombuffer(np.uint32(s.type).tobytes(), dtype=np.float32)[0])
            surf_flat[base + 1] = np.float32(np.frombuffer(np.uint32(s.boundary_condition).tobytes(), dtype=np.float32)[0])

            if s.type == ST.CYLINDER_Z:
                # coefficients = (x0, y0, radius, 0.0)
                # Shader expects: [2]=cx=x0, [3]=cy=y0, [4]=cz=0, [5]=cw=radius
                surf_flat[base + 2] = s.coefficients[0]  # x0
                surf_flat[base + 3] = s.coefficients[1]  # y0
                surf_flat[base + 4] = 0.0                # cz unused
                surf_flat[base + 5] = s.coefficients[2]  # radius → cw
            elif s.type == ST.SPHERE:
                # coefficients = (x0, y0, z0, radius) — hypothetical
                # Shader expects: [2]=cx, [3]=cy, [4]=cz, [5]=cw=radius
                surf_flat[base + 2] = s.coefficients[0]
                surf_flat[base + 3] = s.coefficients[1]
                surf_flat[base + 4] = s.coefficients[2]
                surf_flat[base + 5] = s.coefficients[3]
            else:
                # Planes: coefficients map directly
                surf_flat[base + 2] = s.coefficients[0]
                surf_flat[base + 3] = s.coefficients[1]
                surf_flat[base + 4] = s.coefficients[2]
                surf_flat[base + 5] = s.coefficients[3]
            surf_flat[base + 6] = 0.0
            surf_flat[base + 7] = 0.0

        # Pack cells: 4 uints per cell
        cell_surfs_list = []
        n_cells = len(cells)
        cell_flat = np.zeros(n_cells * 4, dtype=np.uint32)
        offset = 0
        for i, c in enumerate(cells):
            base = i * 4
            cell_flat[base + 0] = c.material_index
            cell_flat[base + 1] = len(c.surfaces)
            cell_flat[base + 2] = offset
            cell_flat[base + 3] = 0
            for surf_idx, sense in c.surfaces:
                cell_surfs_list.append((surf_idx, sense))
            offset += len(c.surfaces)

        # Pack cell-surfaces: 2 ints per entry
        cs_flat = np.zeros(len(cell_surfs_list) * 2, dtype=np.int32)
        for i, (surf_idx, sense) in enumerate(cell_surfs_list):
            cs_flat[i * 2 + 0] = surf_idx
            cs_flat[i * 2 + 1] = sense

        self.surfaces_flat = surf_flat
        self.cells_flat = cell_flat
        self.cell_surfs_flat = cs_flat
        self.num_cells = n_cells
        self.num_surfaces = n_surfs

        # Create GPU buffers for geometry once (reused every batch)
        self._surf_buf = self.engine.make_buffer_from_numpy(surf_flat)
        self._cells_buf = self.engine.make_buffer_from_numpy(cell_flat)
        self._cs_buf = self.engine.make_buffer_from_numpy(cs_flat)

    def _build_pincell_raw(self):
        """Build raw pincell geometry (Surface and Cell objects without GPU buffers)."""
        from ..types import BoundaryCondition as BC, SurfaceType as ST

        surfaces = [
            Surface.plane_x(0.0, bc=BC.REFLECTIVE),
            Surface.plane_x(1.26, bc=BC.REFLECTIVE),
            Surface.plane_y(0.0, bc=BC.REFLECTIVE),
            Surface.plane_y(1.26, bc=BC.REFLECTIVE),
            Surface.plane_z(0.0, bc=BC.REFLECTIVE),
            Surface.plane_z(1.0, bc=BC.REFLECTIVE),
            Surface.cylinder_z(0.63, 0.63, 0.54, bc=BC.TRANSMISSIVE),
        ]
        for i, s in enumerate(surfaces):
            s.id = i

        fuel = Cell(material_index=0, surfaces=[
            (0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, -1)
        ])
        moderator = Cell(material_index=6, surfaces=[
            (0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, 1)
        ])
        return surfaces, [fuel, moderator]

    def _build_assembly_raw(self):
        """Build raw 17x17 assembly geometry."""
        from ..types import BoundaryCondition as BC
        from ..geometry import _ASSEMBLY_N, _PIN_PITCH, _FUEL_RADIUS, _assembly_pin_material

        N = _ASSEMBLY_N
        pitch = _PIN_PITCH
        half = pitch / 2.0

        surfaces = []
        # X-planes: 0..17
        for i in range(N + 1):
            x = i * pitch
            bc = BC.REFLECTIVE if (i == 0 or i == N) else BC.TRANSMISSIVE
            surfaces.append(Surface.plane_x(x, bc=bc))
        # Y-planes: 18..35
        for j in range(N + 1):
            y = j * pitch
            bc = BC.REFLECTIVE if (j == 0 or j == N) else BC.TRANSMISSIVE
            surfaces.append(Surface.plane_y(y, bc=bc))
        # Z-planes: 36, 37
        surfaces.append(Surface.plane_z(0.0, bc=BC.REFLECTIVE))
        surfaces.append(Surface.plane_z(1.0, bc=BC.REFLECTIVE))
        # Cylinders: 38..326
        for row in range(N):
            for col in range(N):
                cx = col * pitch + half
                cy = row * pitch + half
                surfaces.append(Surface.cylinder_z(cx, cy, _FUEL_RADIUS, bc=BC.TRANSMISSIVE))

        for idx, s in enumerate(surfaces):
            s.id = idx

        x_base = 0
        y_base = N + 1
        z_lo = 2 * (N + 1)
        z_hi = z_lo + 1
        cyl_base = z_hi + 1

        cells = []
        for row in range(N):
            for col in range(N):
                left_x = x_base + col
                right_x = x_base + col + 1
                bot_y = y_base + row
                top_y = y_base + row + 1
                cyl_idx = cyl_base + row * N + col
                pin_mat = _assembly_pin_material(row, col)
                bbox = [
                    (left_x, +1), (right_x, -1),
                    (bot_y, +1), (top_y, -1),
                    (z_lo, +1), (z_hi, -1),
                ]
                cells.append(Cell(material_index=pin_mat, surfaces=bbox + [(cyl_idx, -1)]))
                cells.append(Cell(material_index=6, surfaces=bbox + [(cyl_idx, +1)]))

        return surfaces, cells

    def _int_as_float(self, val):
        """Encode int32 value as float32 preserving bit pattern."""
        return np.frombuffer(np.int32(val).tobytes(), dtype=np.float32)[0]

    def _allocate_batch_buffers(self):
        """Pre-allocate all reusable GPU buffers for the transport loop."""
        eng = self.engine
        np_ = self.num_particles
        max_fiss = self.num_particles * 3
        n_tally = self.num_cells * self.num_groups

        # Particle SoA input buffers (float32 × num_particles)
        f32_size = np_ * 4
        self._px_buf = eng.make_buffer(f32_size)
        self._py_buf = eng.make_buffer(f32_size)
        self._pz_buf = eng.make_buffer(f32_size)
        self._ux_buf = eng.make_buffer(f32_size)
        self._uy_buf = eng.make_buffer(f32_size)
        self._uz_buf = eng.make_buffer(f32_size)
        self._group_buf = eng.make_buffer(np_ * 4)   # int32
        self._weight_buf = eng.make_buffer(f32_size)

        # Fission bank output buffers
        self._fiss_x_buf = eng.make_buffer(max_fiss * 4)
        self._fiss_y_buf = eng.make_buffer(max_fiss * 4)
        self._fiss_z_buf = eng.make_buffer(max_fiss * 4)
        self._fiss_g_buf = eng.make_buffer(max_fiss * 4)
        self._fiss_cnt_buf = eng.make_buffer(4)

        # Tally output buffers
        self._tally_flux_buf = eng.make_buffer(n_tally * 4)
        self._tally_fiss_buf = eng.make_buffer(n_tally * 4)

        # Params buffer (7 floats)
        self._params_buf = eng.make_buffer(7 * 4)

        # Fixed buffer list (order matches shader binding indices 0..25)
        self._all_buffers = [
            self._px_buf, self._py_buf, self._pz_buf,             # 0-2
            self._ux_buf, self._uy_buf, self._uz_buf,             # 3-5
            self._group_buf, self._weight_buf,                     # 6-7
            self._fiss_x_buf, self._fiss_y_buf,                   # 8-9
            self._fiss_z_buf, self._fiss_g_buf, self._fiss_cnt_buf,  # 10-12
            self._sigma_t_buf, self._sigma_s_buf, self._sigma_a_buf,  # 13-15
            self._sigma_f_buf, self._nu_sf_buf, self._chi_buf, self._is_fiss_buf,  # 16-19
            self._surf_buf, self._cells_buf, self._cs_buf,        # 20-22
            self._tally_flux_buf, self._tally_fiss_buf,           # 23-24
            self._params_buf,                                      # 25
        ]

    def run(self, geometry_type='pincell', verbose=False):
        """Run k-eigenvalue power iteration.

        Args:
            geometry_type: 'pincell' or 'assembly'
            verbose: Print every batch if True

        Returns:
            dict with simulation results
        """
        self._prepare_geometry(geometry_type)

        # Initial source
        source = self.source_sampler_fn(self.num_particles, self.num_groups)

        # Extract source into SoA arrays
        source_x = np.array([s[0][0] for s in source], dtype=np.float64)
        source_y = np.array([s[0][1] for s in source], dtype=np.float64)
        source_z = np.array([s[0][2] for s in source], dtype=np.float64)
        source_g = np.array([s[1] for s in source], dtype=np.int32)

        max_fission_sites = self.num_particles * 3
        n_tally = self.num_cells * self.num_groups
        batch_keff = []
        entropy_history = []
        k_eff = 1.0

        # Pre-allocate all GPU buffers once (zero-allocation batch loop)
        self._allocate_batch_buffers()

        # Pre-encode static params (fields 0-4, 6 are constant across batches)
        params = np.zeros(7, dtype=np.float32)
        params[0] = self._int_as_float(self.num_particles)
        params[1] = self._int_as_float(self.num_groups)
        params[2] = self._int_as_float(self.num_cells)
        params[3] = self._int_as_float(self.num_surfaces)
        params[6] = self._int_as_float(max_fission_sites)

        # Reusable numpy arrays for direction sampling
        weight_arr = np.ones(self.num_particles, dtype=np.float32)

        print(f"Persistent kernel simulation ({geometry_type})")
        print(f"  Particles:  {self.num_particles}")
        print(f"  Batches:    {self.num_batches} ({self.num_inactive} inactive)")
        print(f"  Cells:      {self.num_cells}")
        print(f"  Surfaces:   {self.num_surfaces}")
        print(f"  GPU:        {self.engine.gpu_name}")
        print()

        wall_start = time.time()

        for batch_idx in range(self.num_batches):
            is_active = batch_idx >= self.num_inactive
            eng = self.engine

            # Write particle positions into pre-allocated buffers
            eng.write_buffer(self._px_buf, source_x.astype(np.float32))
            eng.write_buffer(self._py_buf, source_y.astype(np.float32))
            eng.write_buffer(self._pz_buf, source_z.astype(np.float32))

            # Isotropic random directions
            cos_theta = np.random.uniform(-1, 1, self.num_particles).astype(np.float32)
            sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2)).astype(np.float32)
            phi = np.random.uniform(0, 2*np.pi, self.num_particles).astype(np.float32)
            eng.write_buffer(self._ux_buf, (sin_theta * np.cos(phi)).astype(np.float32))
            eng.write_buffer(self._uy_buf, (sin_theta * np.sin(phi)).astype(np.float32))
            eng.write_buffer(self._uz_buf, cos_theta)

            eng.write_buffer(self._group_buf, source_g.astype(np.int32))
            eng.write_buffer(self._weight_buf, weight_arr)

            # Update per-batch params (k_eff and batch_seed change)
            params[4] = np.float32(k_eff)
            params[5] = self._int_as_float(batch_idx)
            eng.write_buffer(self._params_buf, params)

            # Zero only the fission counter (4 bytes).
            # Fission bank entries are written at kernel-assigned indices so stale
            # data is never read.  Tallies are not used on the Python side currently.
            eng.zero_buffer(self._fiss_cnt_buf)

            # SINGLE GPU DISPATCH -- entire batch in one kernel call
            eng.dispatch_transport(self.num_particles, self._all_buffers)

            # Read fission count
            fiss_count = int(eng.read_buffer(self._fiss_cnt_buf, dtype=np.uint32, count=1)[0])

            # Batch k-eff
            batch_k = fiss_count / self.num_particles if fiss_count > 0 else 0.0

            if is_active:
                batch_keff.append(batch_k)

            # Shannon entropy and source resampling
            if fiss_count > 0:
                fx = eng.read_buffer(self._fiss_x_buf, dtype=np.float32, count=fiss_count)
                fy = eng.read_buffer(self._fiss_y_buf, dtype=np.float32, count=fiss_count)
                fz = eng.read_buffer(self._fiss_z_buf, dtype=np.float32, count=fiss_count)
                fg = eng.read_buffer(self._fiss_g_buf, dtype=np.int32, count=fiss_count)

                entropy = self._shannon_entropy(fx, fy, fz)
                entropy_history.append(entropy)

                # Resample source from fission bank
                indices = np.random.choice(fiss_count, size=self.num_particles, replace=True)
                source_x = fx[indices].astype(np.float64)
                source_y = fy[indices].astype(np.float64)
                source_z = fz[indices].astype(np.float64)
                source_g = fg[indices].copy()
            else:
                entropy_history.append(0.0)
                source = self.source_sampler_fn(self.num_particles, self.num_groups)
                source_x = np.array([s[0][0] for s in source], dtype=np.float64)
                source_y = np.array([s[0][1] for s in source], dtype=np.float64)
                source_z = np.array([s[0][2] for s in source], dtype=np.float64)
                source_g = np.array([s[1] for s in source], dtype=np.int32)

            k_eff = float(batch_k) if batch_k > 0 else k_eff

            # Print progress
            if verbose or batch_idx % 10 == 0 or batch_idx == self.num_batches - 1:
                status = "Active  " if is_active else "Inactive"
                print(f"Batch {batch_idx+1:3d}/{self.num_batches} [{status}] "
                      f"k-eff = {batch_k:.5f}  fission = {fiss_count}")

        wall_time = time.time() - wall_start
        total_particles = self.num_particles * self.num_batches
        particles_per_sec = total_particles / wall_time

        # Statistics
        k_stats = keff_statistics(batch_keff)

        result = {
            'k_eff': k_stats[0],
            'k_eff_std': k_stats[1],
            'k_eff_ci95': k_stats[2],
            'wall_time': wall_time,
            'particles_per_sec': particles_per_sec,
            'total_particles': total_particles,
            'num_active_batches': self.num_batches - self.num_inactive,
            'gpu_name': self.engine.gpu_name,
            'batch_keff_history': batch_keff,
            'entropy_history': entropy_history,
            'geometry_type': geometry_type,
            'num_particles': self.num_particles,
            'num_batches': self.num_batches,
            'reference_keff': self.reference_keff,
        }

        self._print_summary(result)
        return result

    def _shannon_entropy(self, fx, fy, fz, n_bins=8):
        """Compute Shannon entropy of fission site spatial distribution."""
        if len(fx) == 0:
            return 0.0

        # Use numpy histogramdd for fully vectorized 3D binning
        sample = np.column_stack((fx, fy, fz))
        hist, _ = np.histogramdd(sample, bins=n_bins)

        total = hist.sum()
        if total <= 0:
            return 0.0

        probs = hist.flatten() / total
        # Vectorized entropy: -sum(p * log2(p)) for p > 0
        mask = probs > 0
        return float(-np.sum(probs[mask] * np.log2(probs[mask])))

    def _print_summary(self, result):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("PERSISTENT KERNEL SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Geometry:    {result['geometry_type']}")
        print(f"  k-eff =      {result['k_eff']:.5f} +/- {result['k_eff_std']:.5f}")
        lo, hi = result['k_eff_ci95']
        print(f"  95% CI:      [{lo:.5f}, {hi:.5f}]")
        ref = result['reference_keff']
        delta_pcm = (result['k_eff'] - ref) * 1e5
        print(f"  Reference:   {ref:.5f}  (delta = {delta_pcm:+.1f} pcm)")
        print(f"\n  Performance: {result['particles_per_sec']:.0f} particles/sec")
        print(f"  Wall time:   {result['wall_time']:.2f} sec")
        print(f"  GPU:         {result['gpu_name']}")
        print("=" * 60)
