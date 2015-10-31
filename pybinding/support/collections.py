import numpy as np
from matplotlib.collections import Collection, allow_rasterization


# noinspection PyAbstractClass
class CircleCollection(Collection):
    """Custom circle collection

    The default matplotlib `CircleCollection` creates circles based on their
    area in screen units. This class uses the radius in data units. It behaves
    like a much faster version of a `PatchCollection` of `Circle`.
    The implementation is similar to `EllipseCollection`.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        from matplotlib import path, transforms
        self.radius = np.atleast_1d(radius)
        self._paths = [path.Path.unit_circle()]
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))

    def _set_transforms(self):
        ax = self.axes
        self._transforms = np.zeros((self.radius.size, 3, 3))
        self._transforms[:, 0, 0] = self.radius * ax.bbox.width / ax.viewLim.width
        self._transforms[:, 1, 1] = self.radius * ax.bbox.height / ax.viewLim.height
        self._transforms[:, 2, 2] = 1

    @allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)


class Circle3DCollection(CircleCollection):
    def __init__(self, radius, zs=0, zdir='z', depthshade=True, **kwargs):
        super().__init__(radius, **kwargs)
        self._depthshade = depthshade
        self.set_3d_properties(zs, zdir)
        self._A0 = self._A

    def set_array(self, array):
        self._A0 = array
        super().set_array(array)

    def set_3d_properties(self, zs, zdir):
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = list(zip(*offsets))
        else:
            xs = []
            ys = []

        from mpl_toolkits.mplot3d.art3d import juggle_axes
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._facecolor3d = self.get_facecolor()
        self._edgecolor3d = self.get_edgecolor()

    def do_3d_projection(self, renderer):
        from mpl_toolkits.mplot3d import proj3d
        from mpl_toolkits.mplot3d.art3d import zalpha
        from matplotlib import colors as mcolors

        # transform and sort in z direction
        v = np.array(proj3d.proj_transform_clip(*self._offsets3d, M=renderer.M)[:3])
        idx = v[2].argsort()[::-1]
        vzs = v[2, idx]

        self.set_offsets(v[:2, idx].transpose())
        super().set_array(self._A0[idx])

        fcs = zalpha(self._facecolor3d, vzs) if self._depthshade else self._facecolor3d
        fcs = mcolors.colorConverter.to_rgba_array(fcs, self._alpha)
        self.set_facecolors(fcs)

        ecs = zalpha(self._edgecolor3d, vzs) if self._depthshade else self._edgecolor3d
        ecs = mcolors.colorConverter.to_rgba_array(ecs, self._alpha)
        self.set_edgecolors(ecs)

        return min(vzs) if vzs.size > 0 else np.nan
