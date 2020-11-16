import os

import numpy as np

from devito import Grid, SubDomain, Function, Constant, warning, mmin, mmax

__all__ = ['Model']


def initialize_damp(damp, nbpml, spacing, mask=False):
    """
    Initialise damping field with an absorbing PML layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbpml : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """

    phy_shape = damp.grid.subdomains['phydomain'].shape
    data = np.ones(phy_shape) if mask else np.zeros(phy_shape)

    pad_widths = [(nbpml, nbpml) for i in range(damp.ndim)]
    data = np.pad(data, pad_widths, 'edge')

    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)

    assert all(damp._offset_domain[0] == i for i in damp._offset_domain)

    for i in range(damp.ndim):
        for j in range(nbpml):
            # Dampening coefficient
            pos = np.abs((nbpml - j + 1) / float(nbpml))
            val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
            if mask:
                val = -val
            # : slices
            all_ind = [slice(0, d) for d in data.shape]
            # Left slice for dampening for dimension i
            all_ind[i] = slice(j, j+1)
            data[tuple(all_ind)] += val/spacing[i]
            # right slice for dampening for dimension i
            all_ind[i] = slice(data.shape[i]-j, data.shape[i]-j+1)
            data[tuple(all_ind)] += val/spacing[i]

    initialize_function(damp, data, 0)


def initialize_function(function, data, nbpml, pad_mode='edge'):
    """
    Initialize a `Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this function.

    Parameters
    ----------
    function : Function
        The initialised object.
    data : ndarray
        The data array used for initialisation.
    nbpml : int
        Number of PML layers for boundary damping.
    pad_mode : str or callable, optional
        A string or a suitable padding function as explained in :func:`numpy.pad`.
    """
    pad_widths = [(nbpml + i.left, nbpml + i.right) for i in function._size_halo]
    data = np.pad(data, pad_widths, pad_mode)
    function.data_with_halo[:] = data


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbpml):
        super(PhysicalDomain, self).__init__()
        self.nbpml = nbpml

    def define(self, dimensions):
        return {d: ('middle', self.nbpml, self.nbpml) for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbpml=20,
                 dtype=np.float32, subdomains=(), comm=None):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with PML to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbpml) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbpml)
        subdomains = subdomains + (phydomain, )
        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         subdomains=subdomains, comm=comm)

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self._physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class Model(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in m/s
    nbpml : int, optional
        The number of PML layers for boundary damping.
    dtype : np.float32 or np.float64
        Defaults to 32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.

    The `Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    m : array_like or float
        The square slowness of the wave.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, nbpml=20,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 subdomains=(), comm=None, **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbpml, dtype,
                                    subdomains, comm)

        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = Function(name="m", grid=self.grid, space_order=space_order)
        else:
            self.m = Constant(name="m", value=1/vp**2)
        self._physical_parameters = ('m',)
        # Set model velocity, which will also set `m`
        self.vp = vp

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                self._physical_parameters += ('epsilon',)
                self.epsilon = Function(name="epsilon", grid=self.grid)
                initialize_function(self.epsilon, 1 + 2 * epsilon, self.nbpml)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if mmax(self.epsilon) > 0:
                    self.scale = np.sqrt(mmax(self.epsilon))
            else:
                self.epsilon = 1 + 2 * epsilon
                self.scale = epsilon
        else:
            self.epsilon = 1

        if delta is not None:
            if isinstance(delta, np.ndarray):
                self._physical_parameters += ('delta',)
                self.delta = Function(name="delta", grid=self.grid)
                initialize_function(self.delta, np.sqrt(1 + 2 * delta), self.nbpml)
            else:
                self.delta = delta
        else:
            self.delta = 1

        if theta is not None:
            if isinstance(theta, np.ndarray):
                self._physical_parameters += ('theta',)
                self.theta = Function(name="theta", grid=self.grid,
                                      space_order=space_order)
                initialize_function(self.theta, theta, self.nbpml)
            else:
                self.theta = theta
        else:
            self.theta = 0

        if phi is not None:
            if self.grid.dim < 3:
                warning("2D TTI does not use an azimuth angle Phi, ignoring input")
                self.phi = 0
            elif isinstance(phi, np.ndarray):
                self._physical_parameters += ('phi',)
                self.phi = Function(name="phi", grid=self.grid, space_order=space_order)
                initialize_function(self.phi, phi, self.nbpml)
            else:
                self.phi = phi
        else:
            self.phi = 0

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * mmin(self.spacing) / (self.scale*mmax(self.vp)))
        return 1e-9 * int(1e9 * dt)

    @classmethod
    def critical_dx(cls, vmin, fmax):
        """
        Critical computational spatial step value
        """

        dx = vmin / (5*fmax)
        return 1e-5 * int(1e5 * dx)

    @property
    def vp(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        self._vp = vp

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            initialize_function(self.m, 1 / (self.vp * self.vp), self.nbpml)
        else:
            self.m.data = 1 / vp**2
