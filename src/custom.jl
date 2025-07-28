#=******************************************************************************
TENSOR/CARTESIAN PRODUCT BOX DOMAIN - CUSTOM GRIDDING
******************************************************************************=#
"""
    CustomTensorDomain{D}

A type to represent the rectangular domain produced by Cartesian product of D
custom gridded spaces. Each dimension can have its own grid, which can be
non-uniform and non-equidistant. The grids must be provided as a tuple of
`AbstractVector`s, where each vector represents the grid points for that
dimension.

## Fields
- `dimnames::NTuple{D,Symbol}`: Names for each dimension.
- `lb::SV64{D}`: Lower bounds for each dimension.
- `ub::SV64{D}`: Upper bounds for each dimension.
- `Ns::NTuple{D,Int}`: Number of points in each dimension.
- `grids::NTuple{D,Vector{Float64}}`: The actual grids for each dimension.

## Overloaded Base methods
- The same as `TensorDomain{D}`

## Overloaded Statistics methods
- None

## Example
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
"""
struct CustomTensorDomain{D} <: AbstractBoxDomain{D}
    dimnames::NTuple{D,Symbol}
    lb      ::SV64{D}
    ub      ::SV64{D}
    Ns      ::NTuple{D,Int}
    grids   ::NTuple{D,Vector{Float64}}

    function CustomTensorDomain(
        grids::Tuple{Vararg{AbstractVector}} ;
        dimnames::Union{Nothing, Iterable} = nothing,
    )
        d = length(grids)
        @assert d > 0 "CustomTensorDomain must have at least one dimension"
        
        for g in grids
            @assert length(g) > 0 "Each grid must have at least one point"
            @assert all(isfinite, g) "All grid points must be finite"
            @assert issorted(g) "Grid points must be sorted in ascending order"
            @assert allunique(g) "Grid points must be unique"
        end

        lb = SV64{d}(minimum.(grids))
        ub = SV64{d}(maximum.(grids))
        Ns = NTuple{d}(length.(grids))

        _dnames = if isnothing(dimnames)
            Tuple([ Symbol(:x,i) for i in 1:d ])
        else
            @assert length(dimnames) == d "dimnames size mismatch"
            Tuple([ Symbol(dname) for dname in dimnames ])
        end

        new{d}(
            _dnames,
            lb,
            ub,
            Ns,
            grids
        )
    end
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, ctd::CustomTensorDomain{D}) where D
    println(io, "CustomTensorDomain{$D}")
    for j in 1:D
        _nm = ctd.dimnames[j]
        _lb = round(ctd.lb[j], digits=3)
        _ub = round(ctd.ub[j], digits=3)
        _ns = ctd.Ns[j]

        println(io, "  $(_nm): [$(_lb), $(_ub)], #nodes = $(_ns)")
    end
    println(io, "  #points: $(ctd.Ns |> prod)")
end
# ------------------------------------------------------------------------------
"""
    CustomTensorDomain(td::TensorDomain{D})

Construct a `CustomTensorDomain` from a `TensorDomain`. The two domains are in
fact identical, but converting to `CustomTensorDomain` allows for custom
gridding in each dimension later.
"""
function CustomTensorDomain(
    td::TensorDomain{D}, 
    dimnames::Iterable{Symbol} = td.dimnames
) where D
    CustomTensorDomain(
        collect(td), 
        dimnames = dimnames
    )
end
# ------------------------------------------------------------------------------
function BoxDomain(ctd::CustomTensorDomain)
    return BoxDomain(ctd.lb, ctd.ub, dimnames = ctd.dimnames)
end





# ------------------------------------------------------------------------------
function Base.getindex(
    ctd::CustomTensorDomain{D},
    i::Int
)::Vector{Float64} where D
    @assert 1 <= i <= D "index out of range"
    return ctd.grids[i]
end
function Base.getindex(
	ctd::CustomTensorDomain{D}, 
    i2j::UnitRange{Int}
)::CustomTensorDomain where D
	# return a sub-domain
    return CustomTensorDomain(
    	ctd.grids[i2j],
        dimnames = ctd.dimnames[i2j]
    )
end
function Base.getindex(
    ctd::CustomTensorDomain{D},
    i2j::Vector{Int}
)::CustomTensorDomain where D
    # return a sub-domain
    return CustomTensorDomain(
        ctd.grids[i2j],
        dimnames = ctd.dimnames[i2j]
    )
end
function Base.getindex(
    ctd   ::CustomTensorDomain{D},
    dname::Symbol
)::Vector{Float64} where D
    for i in 1:D
        if ctd.dimnames[i] == dname
            return ctd.grids[i]
        end
    end
    throw(ArgumentError("dimension name $dname not found"))
end
function Base.getindex(
    ctd::CustomTensorDomain{D}, 
    sub::CartesianIndex{D}
)::Vector{Float64} where D
    return [ctd[i][sub[i]] for i in 1:D]
end
# ------------------------------------------------------------------------------
function Iterators.product(ctd::CustomTensorDomain{D}) where D
    return Iterators.product(ctd.grids...)
end
# ------------------------------------------------------------------------------
function Base.collect(
    ctd::CustomTensorDomain{D}
)::NTuple{D,Vector{Float64}} where D
    ctd.grids
end
# ------------------------------------------------------------------------------
function Base.stack(ctd::CustomTensorDomain{D}) where D
    xStack = stack(ctd |> Iterators.product |> collect |> vec, dims = 1)
    return NamedArray(
        xStack,
        names = (
            1:size(xStack,1),
            ctd.dimnames |> collect,
        ),
        dimnames = ("Node", "Dimension")
    )
end

# ------------------------------------------------------------------------------
function Base.Array(ctd::CustomTensorDomain{D}) where D
    return Array{Float64,D}(undef, ctd.Ns...)
end
function Base.zeros(ctd::CustomTensorDomain{D}) where D
    return zeros(Float64, ctd.Ns...)
end
function Base.ones(ctd::CustomTensorDomain{D}) where D
    return ones(Float64, ctd.Ns...)
end

# ------------------------------------------------------------------------------
function Base.length(ctd::CustomTensorDomain{D}) where D
    return prod(ctd.Ns)
end
function Base.size(ctd::CustomTensorDomain{D}) where D
    return ctd.Ns |> Tuple
end

# ------------------------------------------------------------------------------
function Base.CartesianIndices(
    ctd::CustomTensorDomain{D}
)::CartesianIndices{D} where D
    return CartesianIndices(ctd.Ns)
end
function Base.LinearIndices(
    ctd::CustomTensorDomain{D}
)::LinearIndices{D} where D
    return LinearIndices(ctd.Ns)
end

# ------------------------------------------------------------------------------
"""
    map(
        f::Function, 
        ctd::TensorDomain{D}
    )::Array{Float64,D}

Map a function `f(x):R^D -> R` over the grid points of the tensor domain `td`.
Returns a D-dim array of the same size as the array created by `Array(td)`.
The function `f` is supposed to be able to receive a single parameter `x` of
type `NTuple{D,Float64}` and return a single real value that can be converted
to `Float64`.
"""
function Base.map(
    f ::Function,
    ctd::CustomTensorDomain{D}
)::Array{Float64,D} where D
    return map(f, Iterators.product(ctd))
end
# ------------------------------------------------------------------------------
"""
    map!(
        f::Function, 
        out::Array{Float64,D}, 
        ctd::CustomTensorDomain{D}
    )

Apply a function `f(x):R^D -> R` over the grid points of the tensor domain `td`,
and store the results in the pre-allocated array `out`. The function `f` is
supposed to be able to take a single parameter `x` of type `NTuple{D,Float64}`
and return a single real value that can be converted to `Float64`.
"""
function Base.map!(
    f  ::Function,
    out::Array{Float64,D},
    ctd::CustomTensorDomain{D}
) where D
    return map!(f, out, Iterators.product(ctd) |> collect)
end
# ------------------------------------------------------------------------------
"""
    locate(
        x ::AbstractVector, 
        ctd::CustomTensorDomain{D}
    )::Vector{NTuple{2,Int}} where D

Locates a D-dimensional point `x` in the gridded space `ctd`. For dimension `d`,
this function finds the two neighbors (along dimension `d`) nodes `ctd[d][i]` 
and `ctd[d][j]` such that `ctd[d][i] <= x[i] <= ctd[d][j]` brackets the element 
`x[i]`. For `x[i]` locates exactly on grid nodes, `j == i`, otherwise `j = i+1`.
For `x` that falls outside the box domain, define the bracket as `(0,1)` or 
`(ctd[d][end],ctd[d][end])`.

Returns two Int vectors of `i` and `j` indices for each dimension respectively.
"""
function locate(
    x ::AbstractVector, 
    ctd::CustomTensorDomain{D}
)::Vector{NTuple{2,Int}} where D
    @assert x ∈ ctd "x is outside the tensor box domain."

    return [
        (
            searchsortedlast(ctd[d], x[d]),
            searchsortedfirst(ctd[d], x[d]),
        )
        for d in 1:D
    ]
end
# ------------------------------------------------------------------------------
"""
    neighbors(x::AbstractVector, ctd::CustomTensorDomain{D}) where D

Find all the nearest on-grid neighbor points for point `x` in the tensor-defined
box domain `ctd`. The number of the neighbor points ranges from `D` (if all dime
nsion elements locate exactly on the grid points) to `2^D`.

Returns a vector of `D`-length vectors of integer indices in `ctd` grids.

## Tips
- to get the values of each neighbor nodes, do `ctd[res[i] |> CartesianIndex]`,
where `res = bdm.neighbors(x,ctd)` and `i` is the `i`-th neighbor saved in `res`
"""
function neighbors(x::AbstractVector, ctd::CustomTensorDomain{D}) where D
    return Iterators.product(locate(x, ctd)...) |> unique
end




# TODO: other methods, check `TensorDomain` for reference

















