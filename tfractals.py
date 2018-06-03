#! encoding=utf8
from __future__ import absolute_import, unicode_literals, print_function
import sys
import argparse
import threading
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from PIL import Image

"""

TODO:
- Allow for non square images. (There shouldn't be a lot more to do)
- Arguments to select devices


"""


mapping = []
mapping.append((0, 0, 0))
mapping.append((66, 30, 15))
mapping.append((25, 7, 26))
mapping.append((9, 1, 47))
mapping.append((4, 4, 73))
mapping.append((0, 7, 100))
mapping.append((12, 44, 138))
mapping.append((24, 82, 177))
mapping.append((57, 125, 209))
mapping.append((134, 181, 229))
mapping.append((211, 236, 248))
mapping.append((241, 233, 191))
mapping.append((248, 201, 95))
mapping.append((255, 170, 0))
mapping.append((204, 128, 0))
mapping.append((153, 87, 0))
mapping.append((106, 52, 3))
mapping.append((0, 0, 0))

COLORS = np.array(mapping)
N_WORKER_THREADS = 2


def load_image(infilename) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def mandelbrot_set(c, zn):
    real_part = tf.expand_dims(
            tf.square(zn[:, 0]) - tf.square(zn[:, 1]) + c[:, 0], -1)
    im_part = tf.expand_dims(2*zn[:, 0]*zn[:, 1] + c[:, 1], -1)
    return tf.concat([real_part, im_part], axis=1)


def recurse_fct(fct, compute_orbits=False):
    def recurse(c, orbits, idx):
        if compute_orbits:
            zn = orbits[:, -1, :]
        else:
            zn = orbits
        norms = tf.reduce_sum(tf.square(zn), axis=-1, keepdims=True)
        less_than_four = tf.cast(tf.less(norms, 4), dtype=tf.int32)
        idx = tf.add(idx, less_than_four)
        znp1 = tf.clip_by_value(fct(c, zn), -100, 100)
        if compute_orbits:
            orbits = tf.concat([orbits, tf.expand_dims(znp1, axis=1)], axis=1)
            return [c, orbits, idx]
        else:
            return [c, znp1, idx]
    return recurse


class MandelbrotCell(tf.contrib.rnn.RNNCell):

    def __init__(self, fct, **kwargs):
        super(MandelbrotCell, self).__init__(**kwargs)
        self.fct = fct

    @property
    def state_size(self):
        return 2

    @property
    def output_size(self):
        return 2

    def __call__(self, inputs, state, scope=None):
        znp1 = self.fct(inputs, state)
        return znp1, znp1


def build_rnn(fct, n_steps=10):
    cell = MandelbrotCell(fct)
    input_tensor = tf.placeholder("float32", shape=(None, 2))
    input_full = tf.tile(tf.expand_dims(input_tensor, 1), [1, n_steps, 1])
    initial_state = cell.zero_state(
        tf.shape(input_tensor)[0], dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell, input_full,
                                       initial_state=initial_state,
                                       dtype=tf.float32)
    norms = tf.reduce_sum(tf.square(outputs), axis=-1)
    return input_tensor, norms


def build_mandelbrot_loop(n_steps=10):
    c = tf.placeholder("float32", shape=(None, 2))  # c is the input
    z0 = tf.zeros_like(c)
    escapes, znp1 = build_loop(mandelbrot_set, z0, c, n_steps=n_steps)
    return c, escapes, znp1


def build_julia_loop(c_base, n_steps=10, compute_orbits=True):
    c = tf.get_variable('c', initializer=c_base)
    z0 = tf.placeholder("float32", shape=(None, 2))  # z0 is the input
    escapes, orbits = build_loop(mandelbrot_set, z0, c, n_steps=n_steps,
                                 compute_orbits=compute_orbits)
    return z0, escapes, orbits


def build_loop(fct, z0, c, n_steps=10, compute_orbits=False):
    escapes = tf.zeros((tf.shape(z0)[0], 1), dtype="int32")
    if compute_orbits:
        orbits = tf.expand_dims(z0, axis=1)
        shape_invariants = [c.get_shape(), tf.TensorShape([None, None, 2]),
                           escapes.get_shape()]
    else:
        orbits = z0
        shape_invariants = None
    _, orbs, esc = tf.while_loop(
        lambda a, b, c: True,
        recurse_fct(fct, compute_orbits=compute_orbits), [c, orbits, escapes],
        parallel_iterations=1,
        maximum_iterations=n_steps,
        shape_invariants=shape_invariants)
    # norms = tf.reduce_sum(tf.square(orbit), axis=-1)
    return esc, orbs


def generate_views(batch_size, x_min, x_max, y_min, y_max, res):
    # if batch_size & (batch_size-1) != 0:
    #     raise ValueError("batch_size must be a power of 2")
    resx, resy = float(res[0]), float(res[1])
    lx = x_max - x_min
    x = 0
    ofsx = x_min
    ly = y_max - y_min
    scalex = lx/resx
    scaley = ly/resy
    while x < resx:
        y, ofsy = 0, y_min
        sx = min(batch_size, resx-x)
        while y < resy:
            sy = min(batch_size, resy-y)
            vx = np.linspace(ofsx, ofsx+sx*scalex, num=sx)
            vy = np.linspace(ofsy, ofsy+sy*scaley, num=sy)
            mx, my = np.meshgrid(vx, vy)
            yield np.concatenate(
                [
                    np.expand_dims(mx, -1),
                    np.expand_dims(my, -1)
                ],
                axis=2
                ).astype("float32").reshape((-1, 2)), (int(resy-y-sy), int(sy), int(x), int(sx))
            ofsy += sy*scaley
            y += sy
        x += sx
        ofsx += sx*scalex


def render_color_gradient_from_escapes(
        viewport, n_escapes, orbits, canvas, n_max, gradient=COLORS):
    n_colors = len(gradient)
    i, di, j, dj = viewport
    n_escapes = (n_escapes.reshape(dj, di)*float(n_colors-1)/(n_max-1)).astype("int")
    n_escapes = np.flip(n_escapes, axis=0)  # don't forget to flip the vertical axis
    colors_to_use = np.concatenate(
            [
                np.expand_dims(np.take(gradient[:, 0], n_escapes), 2),
                np.expand_dims(np.take(gradient[:, 1], n_escapes), 2),
                np.expand_dims(np.take(gradient[:, 2], n_escapes), 2)
            ],
            axis=2
        )
    canvas[i:i+di, j:j+dj] = colors_to_use


def get_trapped_orbit_indices(orbits, position_oti_in_plane):
    print(position_oti_in_plane)
    otix_min, otix_max, otiy_min, otiy_max = position_oti_in_plane
    # step 1
    X = orbits[:, :, :, 0]
    Y = orbits[:, :, :, 1]
    N = np.ones_like(X, dtype="int8")

    N[X < otix_min] = 0
    N[X > otix_max] = 0
    N[Y < otiy_min] = 0
    N[Y > otiy_max] = 0

    N[N > 0] = 1
    N[:, :, -1] = 1

    # step 2
    orbit_trapped = np.argmax(N, axis=-1)
    return orbit_trapped


def get_trapped_coordinates(orbits, orbit_trapped):
    X = orbits[:, :, :, 0]
    Y = orbits[:, :, :, 1]
    # step 3
    stride0 = orbits.shape[1]*orbits.shape[2]
    stride1 = orbits.shape[2]
    orbit_trapped_flat = (orbit_trapped + np.arange(0, orbits.shape[0]).reshape((-1, 1))*stride0)\
                          + np.arange(0, orbits.shape[1])*stride1

    trappedX = np.take(X, orbit_trapped_flat)
    trappedY = np.take(Y, orbit_trapped_flat)
    trappedX[trappedX > 50] = 50
    trappedY[trappedY > 50] = 50
    trappedX[trappedX < -50] = -50
    trappedY[trappedY < -50] = -50
    return trappedX, trappedY


def get_coords_in_image(orbit_trap_image, position_oti_in_plane, trappedX,
                        trappedY):
    otix_min, otix_max, otiy_min, otiy_max = position_oti_in_plane
    # step 4
    Npy, Npx = orbit_trap_image.shape[:2]
    scaleX = (Npx-1)/(otix_max-otix_min)
    scaleY = (Npy-1)/(otiy_max-otiy_min)
    print((scaleX, scaleY))
    j_coord_in_trap = (np.round((trappedX-otix_min)*scaleX)).astype("int32")
    i_coord_in_trap = (np.round((otiy_max-trappedY)*scaleY)).astype("int32")

    # Because we potentially have some points in trappedX and trappedY that are
    # actually outside the trap, we will have indexing errors if we use
    # directly the values returned by the steps above. So, we have to clamp
    # them to a valid range before proceeding.
    j_coord_in_trap[j_coord_in_trap >= Npx] = Npx-1
    i_coord_in_trap[i_coord_in_trap >= Npy] = Npy-1
    j_coord_in_trap[j_coord_in_trap < 0] = 0
    i_coord_in_trap[i_coord_in_trap < 0] = 0
    return i_coord_in_trap, j_coord_in_trap


def get_rbg_from_pixel_coords_in_trap(i_coord_in_trap, j_coord_in_trap,
                                      orbit_trapped, orbit_trap_image, n_max):
    Npy, Npx = orbit_trap_image.shape[:2]
    # step 5
    flat_coord_in_trap = i_coord_in_trap*Npy*3 + j_coord_in_trap*3

    R = np.take(orbit_trap_image, flat_coord_in_trap)
    G = np.take(orbit_trap_image, flat_coord_in_trap+1)
    B = np.take(orbit_trap_image, flat_coord_in_trap+2)

    # now sanitize because some points may be lying out of the valid domain
    # step 6

    condition = (orbit_trapped < n_max-1)
    R = np.expand_dims(np.select([condition], [R], 177), axis=-1)
    G = np.expand_dims(np.select([condition], [G], 197), axis=-1)
    B = np.expand_dims(np.select([condition], [B], 216), axis=-1)
    return R, G, B


def render_orbital_trap(viewport, n_escapes, orbits, canvas, n_max,
                        orbit_trap_image=None, position_oti_in_plane=None):
    """
    n_max designs the number of iteration of the fractal generating function,
    and therefore, the length of our orbital sequences.

    orbits has shape (di, dj, n_max, 2)

    We want to retrieve the first point of each orbit that falls into
    the trap. Here is the strategy
    1 - calculate the boolean matrix idicating if the kth orbit
        point generated by the point represented by the entry i, j is within
        the trap. To be able to know what orbits never entered the trap, we
        also set the last entry for each sequence to True. Points i,j whose
        orbit is not trapped will only have a 1 at position n_max-1.
        Shape: (di, dj, n_max).
    2 - a call to argmax on this matrix will now return a matrix containing
        the positions in the orbits matrix of the first orbit point to fall
        in the trap for each generating point (i, j), and n_max-1 for
        generating points whose orbit never enters the trap. Shape: (di, dj)
    
    Once this is done, we can use the values returned by the argmax (that
    are different from n_max-1) to the retrieve the coordinate of the
    corresponding orbit point, map these plane coordinates to coordinates in
    the orbit trap image, and color with the corresponding pixel. The way we do
    that is:

    3 - Use the indices for the first orbital point in the trap we computed in
        step 4, to retrieve the corresponding X and Y coordinates in matrices.
        We convert values in orbit_trapped to flat coordinates so we can use
        np.take to retrieve the corresponding coordinates in X, and Y

    4 - Now, we convert these plane coordinates into pixel positions in the trap
        image. The pixels corresponding to points that should not be plotted
        (did not fall into the trap, and for which the trapped orbit index is
        n_max-1 (step 3 and 4)) will be removed afterwards.

    5 - Use these pixel coordinates to gather the pixel values the different
        channels. Again, we use flat indexes to use np.take.

    6 - Finally, we use our condition on the value of the orbital index, which
        by construction, must be lesser than n_max-1, to set the pixel values
        for the points outside the trap to a default value.

    7 - Profit
    """
    i, di, j, dj = viewport
    print(orbits.shape)
    assert di*dj == orbits.shape[0]
    orbits = orbits.reshape(dj, di, n_max, 2)
    orbits = np.flip(orbits, axis=0)
    assert orbit_trap_image is not None
    assert position_oti_in_plane is not None
    assert len(orbits.shape) == 4

    print(position_oti_in_plane)

    orbit_trapped = get_trapped_orbit_indices(orbits, position_oti_in_plane)
    trappedX, trappedY = get_trapped_coordinates(orbits, orbit_trapped)

    i_coord_in_trap, j_coord_in_trap = get_coords_in_image(
        orbit_trap_image, position_oti_in_plane, trappedX, trappedY)

    R, G, B = get_rbg_from_pixel_coords_in_trap(
        i_coord_in_trap, j_coord_in_trap, orbit_trapped, orbit_trap_image,
        orbits.shape[2])
    
    # step 7, draw to canvas!

    canvas[i:i+di, j:j+dj] = np.concatenate([R, G, B], axis=-1)
    


def fill_img(canvas, n_max, render_queue, render_function, **render_kwargs):
    # Declarations not necessary but clearer like that
    global COLORS
    global N_WORKER_THREADS

    exited_workers = 0
    while True:
        job = render_queue.get()
        if job is None:
            exited_workers += 1
            render_queue.task_done()
            if exited_workers == N_WORKER_THREADS:
                return
            else:
                continue
        start = time()
        n_escapes, orbits, viewport = job
        render_function(viewport, n_escapes, orbits, canvas, n_max, **render_kwargs)
        render_queue.task_done()
        print("Rendered in {} seconds".format(time()-start))


def exit_all_threads(worker_queue, render_queue):
    global N_WORKER_THREADS
    for _ in range(N_WORKER_THREADS):
        worker_queue.put(None)
        render_queue.put(None)


def worker_loop(device, build_graph, worker_queue, render_queue):
    with tf.device(device):
        input_tensor, escapes, orbits = build_graph()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)) as sess:
            try:
                init = tf.global_variables_initializer()
                sess.run(init)
                while True:
                    job = worker_queue.get()
                    if job is None:
                        worker_queue.task_done()
                        break
                    batch, viewport = job
                    print(viewport)
                    feed_dict = {input_tensor: batch}
                    start = time()
                    res = sess.run([escapes, orbits], feed_dict=feed_dict)
                    print("Executed graph in {} seconds".format(time()-start))
                    print(res[0])
                    print(res[0].shape)
                    # escapes[escapes > 4] = 4
                    # escapes[:, -1] = 4
                    # n_escape = np.argmax(escapes, axis=1)
                    render_queue.put((res[0], res[1], viewport))
                    worker_queue.task_done()
            except Exception as e:
                exit_all_threads(worker_queue, render_queue)
                raise
    render_queue.put(None)


def validate_mandelbrot_arguments(args):
    if args.color_mode == "orbits":
        raise NotImplementedError(
            "Orbits coloring has not been implemented with mandelbrot mode yet.")


def validate_julia_arguments(args):
    assert args.oti is not None
    assert args.poti is not None


def parse_array_arguments(args):
    def _parse(s):
        return tuple([float(x.strip()) for x in s.split(",")])
    args.window = _parse(args.window)
    if args.poti is not None:
        args.poti = _parse(args.poti)
    args.julia_c = _parse(args.julia_c)


def run_mandelbrot(args):
    n_max = args.n
    build_graph = lambda: build_mandelbrot_loop(n_steps=n_max)
    render_function = render_color_gradient_from_escapes
    run(args, build_graph, render_function, {})


def run_julia(args):
    n_max = args.n
    c = np.array([args.julia_c], dtype="float32")
    if args.color_mode == "orbits":
        render_function = render_orbital_trap
        compute_orbits = True
    else:
        render_function = render_color_gradient_from_escapes
        compute_orbits = False

    build_graph = lambda: build_julia_loop(
        c, n_steps=n_max, compute_orbits=compute_orbits)

    oti = load_image(args.oti)
    poti = args.poti
    render_kwargs = {"orbit_trap_image": oti, "position_oti_in_plane": poti}

    run(args, build_graph, render_function, render_kwargs)


def run(args, build_graph, render_function, render_kwargs):
    global N_WORKER_THREADS
    n_max = args.n
    filename = args.filename
    devices = ["/cpu:0"]
    N_WORKER_THREADS = min(N_WORKER_THREADS, len(devices))
    batch_size = args.batch_size
    window = args.window
    res = args.resolution
    img = np.zeros((res, res, 3), dtype="uint8")
    x_min, x_max, y_min, y_max = window
    render_queue = Queue(maxsize=30)
    worker_queue = Queue(maxsize=30)

    # start the render thread
    render_thread = threading.Thread(
        target=fill_img,
        args=(img, n_max+1, render_queue, render_function),
        kwargs=render_kwargs)
    render_thread.start()

    worker_threads = []

    for i in range(N_WORKER_THREADS):
        t = threading.Thread(
                target=worker_loop,
                args=(devices[i], build_graph, worker_queue, render_queue))
        worker_threads.append(t)
        t.start()

    for batch, viewport in generate_views(
            batch_size, x_min, x_max, y_min, y_max, (res, res)):
        worker_queue.put((batch, viewport))

    for _ in range(N_WORKER_THREADS):
        worker_queue.put(None)

    print("Saving into {}".format(filename))
    render_thread.join()
    worker_queue.join()
    plt.imshow(img)
    scipy.misc.imsave(filename, img)
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Create beautiful fractals')
    parser.add_argument('mode', action="store",
                        default="julia",
                        choices=['mandelbrot', 'julia'],
                        help='Generate a Mandelbrot or a Julia set')
    parser.add_argument('filename', action="store",
                        help='The path where to write the result')
    parser.add_argument('-c', '--color-mode', default="escapes",
                        action="store",
                        choices=['escapes', 'orbits'],
                        help='What type of coloring should be applied.')
    parser.add_argument('-r', '--resolution', type=int, default=1024,
                        help="Set the resolution along an axis. The image will"
                             " have r*r pixels.",
                        nargs='?')
    parser.add_argument('--oti', type=str, help="The orbital trap image to use"
                        " when using orbital coloring", default=None,
                        nargs='?')
    parser.add_argument('--poti', type=str,
                        help="The position of the orbital trap image. Format: "
                             "x_min,x_max,y_min,y_max",
                        nargs='?',
                        default="-0.2,0.2,-0.2,0.2")
    parser.add_argument('--window', type=str,
                        help="The position of the rendering window. Format: "
                             "x_min,x_max,y_min,y_max",
                        nargs='?',
                        default="-1.7,1.7,-1.7,1.7")
    parser.add_argument('-n', type=int,
                        help="The number of iterations for the fractal generation",
                        default=20)
    parser.add_argument('-b', '--batch-size', type=int,
                        help="The square root of the batch size to use. Batches"
                             " with b*b points will be generated",
                        default=512)
    parser.add_argument('--julia-c', type=str,
                        help="The offset coefficient (c) in the Julia "
                             "recurrence formula. Format: real_part,im_part",
                        default="-0.8,0.2")

    args = parser.parse_args()

    if args.mode == "julia":
        validate_julia_arguments(args)
    else:
        validate_mandelbrot_arguments(args)
    parse_array_arguments(args)


    if args.mode == "julia":
        run_julia(args)
    else:
        run_mandelbrot(args)
