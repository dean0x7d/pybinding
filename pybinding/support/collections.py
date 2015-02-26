from matplotlib.collections import PatchCollection, allow_rasterization


class CircleCollection(PatchCollection):
    """
    Custom circle collection

    The default matplotlib CircleCollection creates circles based on their
    area in screen units. This class uses the radius in data units. It behaves
    like a much faster version of a PatchCollection of Circles.
    """
    def __init__(self, radius, **kwargs):
        super().__init__([], **kwargs)
        from matplotlib import path, transforms
        self.radius = radius
        self._paths = [path.Path.unit_circle()]
        self.set_transform(transforms.IdentityTransform())

    def _set_transforms(self):
        from matplotlib import transforms
        import numpy as np
        self._transforms = np.zeros((1, 3, 3))
        self._transforms[:, 0, 0] = self.radius
        self._transforms[:, 1, 1] = self.radius
        self._transforms[:, 2, 2] = 1

        m = self.axes.transData.get_affine().get_matrix().copy()
        m[:2, 2:] = 0
        self.set_transform(transforms.Affine2D(m))

    @allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)
