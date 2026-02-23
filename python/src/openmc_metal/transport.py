"""Transport kernel dispatch wrappers."""

import Metal as MTL


class TransportKernels:
    """Manages Metal compute pipelines for transport kernels."""

    def __init__(self, engine):
        self.engine = engine
        self.xs_lookup_pipeline = engine.make_pipeline("xs_lookup")
        self.distance_pipeline = engine.make_pipeline("distance_to_collision")
        self.move_pipeline = engine.make_pipeline("move_particle")

    def dispatch_xs_lookup(self, particles, materials, cells, params,
                           count: int, command_buffer):
        """Encode XS lookup: buffers [0]particles [1]materials [2]cells [3]params."""
        self.engine.dispatch(
            self.xs_lookup_pipeline,
            [particles, materials, cells, params],
            count, command_buffer
        )

    def dispatch_distance_to_collision(self, particles, params,
                                       count: int, command_buffer):
        """Encode distance sampling: buffers [0]particles [1]params."""
        self.engine.dispatch(
            self.distance_pipeline,
            [particles, params],
            count, command_buffer
        )

    def dispatch_move(self, particles, surfaces, cells, cell_surfaces,
                      params, lost_count, count: int, command_buffer):
        """Encode particle move: buffers [0-5] particles,surfaces,cells,cellSurfaces,params,lostCount."""
        self.engine.dispatch(
            self.move_pipeline,
            [particles, surfaces, cells, cell_surfaces, params, lost_count],
            count, command_buffer
        )
