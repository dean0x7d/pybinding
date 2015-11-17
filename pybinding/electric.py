import pybinding as pb

__all__ = ['constant']


def constant(value):
    @pb.onsite_energy_modifier
    def const_onsite(potential):
        return potential + value

    return const_onsite
