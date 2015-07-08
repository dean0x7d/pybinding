from . import modifier

__all__ = ['constant']


def constant(value):
    @modifier.onsite_energy
    def const_onsite(potential):
        return potential + value

    return const_onsite
