#! encoding=utf8
from __future__ import absolute_import, unicode_literals, print_function

import numpy as np

from tfractals import get_trapped_orbit_indices, get_trapped_coordinates, \
                      get_coords_in_image, get_rbg_from_pixel_coords_in_trap


def test_orbital_rendering():
    position_oti_in_plane = -1., 1., -1., 1.
    orbit_trap_image = (np.random.rand(3, 3, 3)*255).astype("int32")
    seq_1 = np.array([[[1.5, 2.5], [1.5, 3.5], [0, 0], [0.5, 0.5]]])  # in the trap
    seq_2 = np.array([[[1.1, 1.2], [1.3, 1.4], [2.0, -1.0], [2.1, 3.0]]])  # out

    print(seq_1.shape)
    trapped_indices = np.array([[2], [3]])

    orbits = np.expand_dims(np.concatenate([seq_1, seq_2], axis=0), axis=1)
    print(orbits.shape)
    assert np.allclose(orbits.shape, [2, 1, 4, 2])

    orbit_trapped = get_trapped_orbit_indices(orbits, position_oti_in_plane)
    print(orbit_trapped)
    print(orbit_trapped.shape)
    assert np.allclose(orbit_trapped, trapped_indices)

    trappedX, trappedY = get_trapped_coordinates(orbits, orbit_trapped)

    assert np.allclose(trappedX[0], orbits[0, :, 2, 0])
    assert np.allclose(trappedX[1], orbits[1, :, 3, 0])
    assert np.allclose(trappedY[0], orbits[0, :, 2, 1])
    assert np.allclose(trappedY[1], orbits[1, :, 3, 1])

    i_coord_in_trap, j_coord_in_trap = get_coords_in_image(
        orbit_trap_image, position_oti_in_plane, trappedX, trappedY)
    print(i_coord_in_trap)
    print(j_coord_in_trap)
    assert np.allclose(i_coord_in_trap[0][0], [1])
    assert np.allclose(j_coord_in_trap[0][0], [1])
    assert np.allclose(i_coord_in_trap[1][0], [0])
    assert np.allclose(j_coord_in_trap[1][0], [2])

    R, G, B = get_rbg_from_pixel_coords_in_trap(
        i_coord_in_trap, j_coord_in_trap, orbit_trapped, orbit_trap_image, 4)
    assert np.allclose(np.concatenate([R[0][0], G[0][0], B[0][0]], axis=-1),
                       orbit_trap_image[1, 1, :])
    assert np.allclose(np.concatenate([R[1][0], G[1][0], B[1][0]], axis=-1), 0)


if __name__ == "__main__":
    test_orbital_rendering()
