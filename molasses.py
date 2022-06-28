#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pylcp as pl


def make_hamiltonian(atom):
    """
    make_hamiltonian(delta, mass)
    return hamiltonian object for the atom.
    To start, a two-level atom is used.

    input:
    atom - atom from pyLCP e.g. pl.atom('41K')

    returns:
    pylcp.Hamiltonian object
    """

    # we want the D2 line.
    # make blocks for the ground (2s_1/2) and excited (4p_3/2)
    # states
    h_g, mu_q_g = pl.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        Ahfs=atom.state[0].Ahfs/atom.state[2].gammaHz, Bhfs=0, Chfs=0,
        muB=1
    )
    h_e, mu_q_e = pl.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs/atom.state[2].gammaHz, 
        Bhfs=atom.state[2].Bhfs/atom.state[2].gammaHz, Chfs=0,
        muB=1
    )
    # make a matrix that couples the manifolds via a dipole interaction.
    dqij_d2 = pl.hamiltonians.dqij_two_hyperfine_manifolds(
        atom.state[0].J, atom.state[2].J, atom.I
    )
    
    hamiltonian = pl.hamiltonian(h_g, h_e, mu_q_g, mu_q_e, dqij_d2)
    hamiltonian.set_mass(atom.mass)
    return hamiltonian


class Molasses:

    def __init__(self, freq, s):
        """
        Molasses(freq, s)

        intializes an optical molasses with frequency and intensity.

        parameters:
        freq - frequency of lasers in units of atomic energy
        s - intensity of lasers
        """
        self.freq = freq
        self.s = s 

    def make_lasers(self):
        """
        make_lasers()
        return laser beams object corresponding to a molasses
        aligned with the x axis.

        input:
        none

        returns:
        pylcp.laserBeams() object
        """
        return pl.laserBeams([
            {'kvec': np.array([1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
            'pol_coord': 'spherical', 'delta': self.freq, 's': self.s},
            {'kvec': np.array([-1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
            'pol_coord': 'spherical', 'delta': self.freq, 's': self.s}
        ], beam_type=pl.infinitePlaneWaveBeam)


def laser_frequency(atom, delta):
    """
    laser_frequnecy(atom, delta)
    gives frequency of laser for D2 line with given detuning

    input:
    atom - pyLCP atom object
    delta - detuning in units of the atomic frequency
    """
    return atom.transition[1].nu + delta


def solve_motion(molasses, atom, t):
    """
    solve_motion(delta, mass, s)
    solve the equations of motion given the paramters
    for the beams and atoms

    input:
    delta - frequency of laser beams
    s - intensity of beams
    t - time of evolution
    """
    laser_beams = molasses.make_lasers()
    hamiltonian = make_hamiltonian(atom)
    # the magnetic field will be zero here!
    mag_field = lambda x: np.zeros(x.shape)
    # generate the equations of motion
    eqn = pl.rateeq(laser_beams, mag_field, hamiltonian)
    eqn.set_initial_position_and_velocity(
        np.zeros(3), np.zeros(3)
    )
    eqn.set_initial_pop_from_equilibrium()
    # TODO: look at hamiltonian matrix for debugging
    # hamiltonian is not coming out for some reason!
    e_field = lambda r, t: laser_beams.electric_field(r, t)
    b_field = lambda r, t: np.zeros((0, 3))
    ham = lambda r, t: hamiltonian.return_full_H(e_field(r, t), b_field(r, t))
    # end debugging
    import pdb; pdb.set_trace()
    # solve and return solution
    soln = eqn.evolve_motion([0, t], random_recoil=False, 
        random_force=False, freeze_axis=[False, True, True])
    return soln


def main():
    atom = pl.atom("41K")
    delta = laser_frequency(atom, 1e-4)
    s = 1e-2
    t = 50.0
    n_atom = 1

    molasses = Molasses(delta, s)

    # simulate the motion of many atoms with random forces
    solns = []
    for j in range(n_atom):
        solns.append(solve_motion(molasses, atom, t))
    
    # find the distributions of final velocities and positions
    v_final = np.zeros(len(solns))
    r_final = np.zeros(len(solns))
    for i in range(len(solns)):
        v_final[i] = norm(solns[i].v[:, -1])
        r_final[i] = norm(solns[i].r[:, -1])
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(r_final)
    ax[0].set_xlabel("r")
    ax[1].hist(v_final)
    ax[1].set_xlabel("v")
    plt.show()


if __name__ == "__main__":
    main()