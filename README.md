# BoxDomains.jl

Defining, materializing, and affines of multi-dimensional box/rectangular spaces. This package aims to facilitate various types of numerical practice.


## Installation

```julia
pkg> add "https://github.com/Clpr/BoxDomains.jl.git#main"
```


## Usage

Consider a rectangular product space defined by total $D$ finite intervals:

$$
\mathbb{X} := \prod_{i=1}^D [ l_i, u_i ]  \subset \mathbb{R}^D
$$

We call it a _box_ or _rectangular_ space, which is very common in numerical analysis such as 
interpolation and quadratures. In practice, one usually has to perform many routine but unavoidable operations such as: bounding a point into the box; linearly affining a point to another box space; generate discrete grid points dynamically; sampling ths box space; indexing each dimension by its label/name rather than manually managing the name-index mapping; ...

In this package, I implements an elegant abstraction of such box spaces and overload many built-in functions which greatly simplifies the data pipeline and analysis. 


So far, this package has implemented two types of box domains:
- `BoxDomain{D}`: the most generic type which does not assume any grid structure or discretization strategies over the domain. It supports many overloaded built-in functions which are interface to all the other derivative structs.
- `TensorDomain{D}`: beyond a generic box space, this struct assumes a Cartesian-product discretization using evenly-spaced gird node points. In addition to the interface functions, it supports much more type-specific functions due to the Cartesian assumption.
- `CustomTensorDomain{D}`: By relaxing the evenly-spaced grid node assumption, this struct allows the users to specify custom nodes for every dimension. Then, it works almost the same as `TensorDomain{D}`.


**BoxDomain{D}**:
```julia
import BoxDomains as bds

# test: shortcut construction, hypercube in D-dim, D = 3 here
domain = bds.BoxDomain(3)

# test: standard construction, allowing dim names (converted to Symbols)
domain = bds.BoxDomain(
    [1.0, 2.0, 3.0], [4.0, 5.0, 7.0], 
    dimnames = ["k", "l", "m"]
)

# test: interface methods
x = [2.0, 3.0, 4.0]

x ∈ domain          # check if x is in the domain
x in domain         # alias
x ∉ domain          # check if x is not in the domain

clamp(x[2], domain, 2) # only the 2nd component
clamp(x, domain)    # the whole point

rand(domain)        # 1 random point in the domain
rand(domain, 10)    # 10 random points in the domain as a 10*D matrix

LinRange(domain, 1, 50) # a LinRange for the 1st dimension, 50 points
LinRange(domain, :k, 50) # indexed by dimension name

bds.centroid(domain)    # the centroid of the domain, (lb + ub) / 2

bds.affine(x, domain, bds.BoxDomain(3)) # affine transform to a hypercube
bds.affine(x, domain, lb = -1, ub = 1) # affine transform to [-1,1]^D

bds.rsg(domain, 4) # create a 2-based lattice regular sparse grid, accuracy 4

merge(domain,domain) # merge 2 domains to a higher-dimensional one; duplicate dimension names are suffixed with "_2"
```

**TensorDomain{D}**:
```julia
# test: shortcut construction
sDomain = bds.TensorDomain([5,6,7]) # 3D hypercube [0,1]^3
sDomain = bds.TensorDomain(
    bds.BoxDomain(3), [5,6,7]
) # convert a BoxDomain to a TensorDomain


# test: standard construction
sDomain = bds.TensorDomain(
    [1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [5, 6, 7], 
    dimnames=["k", "l", "m"]
)

# standard interface methods:
x = [2.0, 3.0, 4.0]
x ∈ sDomain
x in sDomain
x ∉ sDomain
clamp(x[2], sDomain, 2)
clamp(x, sDomain)
rand(sDomain)
rand(sDomain, 10)
LinRange(sDomain, 1, 50)
LinRange(sDomain, :k, 50)
bds.centroid(sDomain)
bds.affine(x, sDomain, bds.TensorDomain([2,3,4]))
bds.affine(x, sDomain, lb = -1, ub = 1)
bds.rsg(sDomain, 4)

# TensorDomain-specific overloaded Base methods

sDomain[1]   # indexing: returns a LinRange of the 1st dimension
sDomain[:k]  # indexing: by dimension name
sDomain[2:3] # slicing: returns a sliced 2-dim TensorDomain by the 2nd, 3rd dim
sDomain[CartesianIndex(1,2,3)] # returns a 3D point in the domain, Cartesian

Iterators.product(sDomain) # creates a product iterator for the domain

collect(sDomain) # returns a D-tuple of LinRange, useful for Interpolations.jl

stack(sDomain) # creates a tensor-stacked NamedMatrix, prod(Ns) x D size

Array(sDomain) # initialize a tensor-stacked size Array, filled by undef/0/1
zeros(sDomain)
ones(sDomain)

merge(domain,domain) # merge 2 domains to a higher-dimensional one; duplicate dimension names are suffixed with "_2"

length(sDomain) # get how many grid points in total
size(sDomain)   # get how many grid points in each dimension as a D-int tuple
ndims(sDomain)  # get the dimensionality of the domain

LinRange(sDomain, 1, 50) # creates a LinRange for the 1st dimension, 50 points

CartesianIndices(sDomain) # creates a CartesianIndices for the domain
LinearIndices(sDomain)    # creates a LinearIndices for the domain

map(sum, sDomain) # map a function over the grid points
map!(sum, zeros(sDomain), sDomain) # map a function over the grid points, store


bds.locate([1,0.5,100], sDomain) # locate a point in the CustomTensorDomain, sandwiched by the indices of the nearest two grid poitns along each dimension
bds.locate(rand(3), sDomain)

bds.neighbors([0.9,0.4,2.0], sDomain) # get the nearest on-grid neighbors of a point in the CustomTensorDomain; denoted by the node indices

```

**CustomTensorDomain{D}**

```julia
import BoxDomains as bds

# convert from a TensorDomain
sDomain = bds.CustomTensorDomain(bds.TensorDomain([4,5,6]))

# standard constructor: manually specify the grids
sDomain = bds.CustomTensorDomain(
    (
        LinRange(0,1,20),
        [-2,0.5,0.9,1.1],
        [0,1,100]
    ),
    dimnames = ["k", "l", "m"]
)

# convert to a BoxDomain
bds.BoxDomain(sDomain)


# standard methods
x = [2.0, 3.0, 4.0]
x ∈ sDomain
x in sDomain
x ∉ sDomain
clamp(x[2], sDomain, 2)
clamp(x, sDomain)
rand(sDomain)
rand(sDomain, 10)
LinRange(sDomain, 1, 50)
LinRange(sDomain, :m, 50)
bds.centroid(sDomain)
bds.affine(x, sDomain, bds.TensorDomain([2,3,4]))
bds.affine(x, sDomain, lb = -1, ub = 1)
bds.rsg(sDomain, 4)


# CustomTensorDomain-specific overloaded Base methods


sDomain[1]   # indexing: returns a LinRange of the 1st dimension
sDomain[:k]  # indexing: by dimension name
sDomain[2:3] # slicing: returns a sliced 2-dim TensorDomain by the 2nd, 3rd dim
sDomain[CartesianIndex(1,2,3)] # returns a 3D point in the domain, Cartesian

Iterators.product(sDomain) # creates a product iterator for the domain

collect(sDomain) # returns a D-tuple of LinRange, useful for Interpolations.jl

stack(sDomain) # creates a tensor-stacked NamedMatrix, prod(Ns) x D size


Array(sDomain) # initialize a tensor-stacked size Array, filled by undef/0/1
zeros(sDomain)
ones(sDomain)

merge(domain,domain) # merge 2 domains to a higher-dimensional one; duplicate dimension names are suffixed with "_2"

length(sDomain) # get how many grid points in total
size(sDomain)   # get how many grid points in each dimension as a D-int tuple
ndims(sDomain)  # get the dimensionality of the domain

LinRange(sDomain, 1, 50) # creates a LinRange for the 1st dimension, 50 points

CartesianIndices(sDomain) # creates a CartesianIndices for the domain
LinearIndices(sDomain)    # creates a LinearIndices for the domain

map(sum, sDomain) # map a function over the grid points
map!(sum, zeros(sDomain), sDomain) # map a function over the grid points, store


bds.locate([1,0.5,100], sDomain) # locate a point in the CustomTensorDomain, sandwiched by the indices of the nearest two grid poitns along each dimension
bds.locate(rand(3), sDomain)

bds.neighbors([0.9,0.4,2.0], sDomain) # get the nearest on-grid neighbors of a point in the CustomTensorDomain; denoted by the node indices

```




Specially, this package implements a convenient method `rsg()` to
create a 2-based regular sparse grid for the given box space. This is especially
useful if one wants to quickly test methods such as orthogonal polynomials.
To unlock the full power and functionalities, please check my another package
`AdaptiveSG.jl`.







## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.













