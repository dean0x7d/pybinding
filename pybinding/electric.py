from . import modifier

__all__ = ['constant']


def constant(value):
    @modifier.onsite_energy
    def const_onsite(potential, x, y, z):
        return potential + value

    return const_onsite
