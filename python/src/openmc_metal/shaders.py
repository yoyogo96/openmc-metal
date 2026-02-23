"""Metal shader source loader."""

from pathlib import Path

# Order matters: Common defines shared types used by all others
SHADER_FILES_ORDER = [
    'Common.metal',
    'Geometry.metal',
    'XSLookup.metal',
    'Transport.metal',
    'Collision.metal',
    'Tally.metal',
]


def load_shader_source(shader_dir: Path | None = None) -> str:
    """Load and concatenate Metal shader source files.

    Tries shader_dir first, then ../../Sources/Shaders/ relative to this file,
    then falls back to embedded string.
    """
    # Try provided directory
    if shader_dir and shader_dir.is_dir():
        return _load_from_dir(shader_dir)

    # Try relative to this file (repo layout: python/src/openmc_metal/ -> Sources/Shaders/)
    repo_shaders = Path(__file__).resolve().parent.parent.parent.parent / 'Sources' / 'Shaders'
    if repo_shaders.is_dir():
        return _load_from_dir(repo_shaders)

    raise FileNotFoundError(
        f"Metal shader directory not found. Tried: {shader_dir}, {repo_shaders}"
    )


def _load_from_dir(shader_dir: Path) -> str:
    """Read and concatenate .metal files in canonical order.

    Handles deduplication of #include headers and constant definitions
    that appear in multiple .metal files (which compile separately in Xcode
    but must be merged into one source string for runtime compilation).
    """
    import re

    parts = []
    header_added = False
    seen_constants: set[str] = set()

    for filename in SHADER_FILES_ORDER:
        filepath = shader_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing shader file: {filepath}")

        content = filepath.read_text()

        # Remove duplicate headers after the first file
        if header_added:
            content = content.replace('#include <metal_stdlib>', '')
            content = content.replace('using namespace metal;', '')
        else:
            header_added = True

        # Remove duplicate constant definitions (e.g. FLOATS_PER_MATERIAL
        # defined in both XSLookup.metal and Collision.metal)
        lines = content.split('\n')
        filtered = []
        for line in lines:
            m = re.match(r'^constant\s+\w+\s+(\w+)\s*=', line)
            if m:
                name = m.group(1)
                if name in seen_constants:
                    continue  # skip duplicate
                seen_constants.add(name)
            filtered.append(line)
        content = '\n'.join(filtered)

        parts.append(f'// === {filename} ===\n{content}')

    source = '\n'.join(parts)
    source = _fix_struct_layouts(source)
    source = _fix_reflection_logic(source)
    return source


def _fix_struct_layouts(source: str) -> str:
    """Fix struct layouts for Python/PyObjC compatibility.

    Metal float3 is 16 bytes (aligned to 16-byte boundary). The original
    Metal/Swift structs add explicit _pad0/_pad1 fields after float3 members,
    which creates alignment gaps (float3=16 + _pad=4 + 12-byte gap = 32 bytes
    per block instead of 16). Removing these explicit pads makes the struct
    layout match the Python struct format, since float3's implicit 4th byte
    serves as the padding.

    Affected structs and their sizes after fix:
      Particle:    144 -> 112 bytes (removes _pad0, _pad1)
      FissionSite:  48 ->  32 bytes (removes _pad)
    """
    import re

    # 1. Remove explicit _pad0 and _pad1 float fields from Particle struct
    #    These duplicate float3's built-in 16-byte padding
    source = re.sub(
        r'\n\s*float\s+_pad0;\s*//[^\n]*',
        '', source
    )
    source = re.sub(
        r'\n\s*float\s+_pad1;\s*//[^\n]*',
        '', source
    )

    # 2. Remove _pad0/_pad1 assignments in kernel code (init_particles, collision)
    source = re.sub(
        r'\n\s*p\._pad[01]\s*=\s*0\.0f;\s*',
        '\n', source
    )

    # 3. Remove explicit _pad float field from FissionSite struct
    source = re.sub(
        r'\n\s*float\s+_pad;\s*\n',
        '\n', source
    )

    # 4. Remove _pad assignment for fission bank entries
    source = re.sub(
        r'\n\s*fissionBank\[idx\]\._pad\s*=\s*0\.0f;\s*\n',
        '\n', source
    )

    return source


def _fix_reflection_logic(source: str) -> str:
    """Fix move_particle kernel for correct boundary handling.

    Two fixes applied:
    1. Reflective BC: move to surface, reflect, bump in reflected direction,
       stay in same cell (prevents particle escaping).
    2. Boundary-distance subtraction: after crossing a boundary, subtract the
       boundary distance from distanceToCollision and loop back to MOVE state
       (not XS_LOOKUP). This prevents infinite zero-distance boundary crossings
       when particles sit on transmissive surfaces, and is the standard MC
       tracking approach (particles carry their remaining collision distance
       across boundaries in the same material).
       XS_LOOKUP is only needed after transmissive crossings where the material
       changes.
    """
    old_boundary_block = """\
    } else {
        // -----------------------------------------------------------
        // Hit boundary -- move to surface + BUMP past it
        // -----------------------------------------------------------
        float d = dBoundary;
        p.position += (d + BUMP) * p.direction;
        p.distanceTraveled = d;

        // Check boundary condition of the crossed surface
        if (crossedSurfIdx >= 0 && uint(crossedSurfIdx) < params.numSurfaces) {
            uint bc = surfaces[crossedSurfIdx].boundaryCondition;

            if (bc == BC_VACUUM) {
                // Particle escapes the problem domain
                p.alive = 0;
                p.event = EVENT_DEAD;
                return;
            } else if (bc == BC_REFLECTIVE) {
                // Reflect direction off the surface normal
                float3 normal = surface_normal(p.position, surfaces[crossedSurfIdx]);
                p.direction = p.direction - 2.0f * dot(p.direction, normal) * normal;
                // Re-normalize to prevent floating-point drift
                p.direction = normalize(p.direction);
            }
        }

        // Find which cell the particle is now in
        int newCell = find_cell(p.position, cells, params.numCells,
                                cellSurfaces, surfaces);
        if (newCell < 0) {
            // Lost particle -- outside all defined cells
            p.alive = 0;
            p.event = EVENT_DEAD;
            atomic_fetch_add_explicit(lostParticleCount, 1, memory_order_relaxed);
            return;
        }
        p.cellIndex = uint(newCell);

        // Need new cross-section lookup in the (possibly different) material
        p.event = EVENT_XS_LOOKUP;
    }"""

    new_boundary_block = """\
    } else {
        // -----------------------------------------------------------
        // Hit boundary
        // -----------------------------------------------------------
        float d = dBoundary;
        // If boundary is essentially at zero distance (re-hit same surface),
        // nudge particle forward to escape the surface and retry
        if (d < 1.0e-6f) {
            p.position += 1.0e-5f * p.direction;
            // Don't change event — will be retried as MOVE next step
            p.event = EVENT_MOVE;
            return;
        }
        p.distanceTraveled = d;
        // Subtract boundary distance from remaining collision distance
        p.distanceToCollision -= d;

        // Check boundary condition of the crossed surface
        if (crossedSurfIdx >= 0 && uint(crossedSurfIdx) < params.numSurfaces) {
            uint bc = surfaces[crossedSurfIdx].boundaryCondition;

            if (bc == BC_VACUUM) {
                p.position += (d + BUMP) * p.direction;
                p.alive = 0;
                p.event = EVENT_DEAD;
                return;
            } else if (bc == BC_REFLECTIVE) {
                // Move exactly to the surface (no bump past it)
                p.position += d * p.direction;
                // Reflect direction off the surface normal
                float3 normal = surface_normal(p.position, surfaces[crossedSurfIdx]);
                p.direction = p.direction - 2.0f * dot(p.direction, normal) * normal;
                p.direction = normalize(p.direction);
                // Bump in the reflected direction (back into the cell)
                p.position += BUMP * p.direction;
                // Stay in same cell, continue moving with remaining d2c
                p.event = EVENT_MOVE;
                return;
            }
        }

        // Transmissive or unknown BC: bump past surface, find new cell
        // Use larger bump (100x BUMP) to avoid getting stuck on curved surfaces
        p.position += (d + 100.0f * BUMP) * p.direction;

        int newCell = find_cell(p.position, cells, params.numCells,
                                cellSurfaces, surfaces);
        if (newCell < 0) {
            p.alive = 0;
            p.event = EVENT_DEAD;
            atomic_fetch_add_explicit(lostParticleCount, 1, memory_order_relaxed);
            return;
        }

        uint oldMat = cells[p.cellIndex].materialIndex;
        p.cellIndex = uint(newCell);
        uint newMat = cells[newCell].materialIndex;

        if (oldMat != newMat) {
            // Material changed: need new XS and new d2c
            p.event = EVENT_XS_LOOKUP;
        } else {
            // Same material: continue with remaining d2c
            p.event = EVENT_MOVE;
        }
    }"""

    if old_boundary_block not in source:
        raise RuntimeError(
            "_fix_reflection_logic: could not find the boundary handling block "
            "in move_particle. The Metal shader source may have changed."
        )

    return source.replace(old_boundary_block, new_boundary_block)
