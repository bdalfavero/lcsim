#!/usr/bin/env python3

import numpy as np
import pylcp as pl


def make_hamiltonian(omega, mass):
    """
    make_hamiltonian(delta, mass)
    return hamiltonian object for the atom.
    To start, a two-level atom is used.

    input:
    delta - excited state energy
    mass - mass of atom

    returns:
    pylcp.Hamiltonian object
    """
    hg = np.array([[0.]]) # ground state hamiltonian
    he = np.array([[-omega]]) # excited state hamiltonian
    mu_q = np.zeros((3, 1, 1))
    d_q = np.zeros((3, 1, 1))
    d_q[1, 0, 0] = 1.0

    return pl.Hamiltonian(hg, he, mu_q, d_q, mass=mass)


def make_lasers(delta, s):
    """
    make_lasers(delta, s)
    return laser beams object

    input:
    delta - laser frequency
    s - intensity

    returns:
    pylcp.laserBeams() object
    """
    return pl.laserBeams([
        {'kvec': np.array([1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
         'pol_coord': 'spherical', 'delta': delta, 's': s},
        {'kvec': np.array([-1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
         'pol_coord': 'spherical', 'delta': delta, 's': s}
    ], beam_type=pl.infinitePlaneWaveBeam)


def solve_motion(energy, mass, delta, s):
    """
    solve_motion(delta, mass, s)
    solve the equations of motion given the paramters
    for the beams and atoms

    input: 
    energy - excited state energy of atom
    mass - mass of atom
    delta - detuning of laser beams
    s - intensity of beams
    """
    laser_beams = make_lasers(delta, s)
    hamiltonian = make_hamiltonian(energy, mass)
    # the magnetic field will be zero here!
    mag_field = lambda x: np.zeros(x.shape)
    # generate the equations of motion
    eqn = pl.rateeq(laser_beams, mag_field, hamiltonian)


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()