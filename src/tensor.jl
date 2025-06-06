export locate

#=******************************************************************************
TENSOR/CARTESIAN PRODUCT BOX DOMAIN
******************************************************************************=#
"""
    TensorDomain{D}

A type to represent the rectangular domain produced by Cartesian product of D 
intervals.

## Fields
- `dimnames::NTuple{D,Symbol}`: names of the dimensions
- `lb::SV64{D}`: lower bounds of the intervals
- `ub::SV64{D}`: upper bounds of the intervals
- `Ns::NTuple{D,Int}`: number of grid points in each dimension

## Overloaded Base methods
- `getindex[]`: returns a LinRange of the i-th dimension
- `collect`: returns a D-tuple of LinRange (works with Interpolations.jl)
- `stack`: returns a tensor-stacked NamedMatrix, prod(Ns) x D
- `Array`: constructs a tensor-stacked size Array, filled by undef, which has
(Ns[1], Ns[2],...,Ns[D]) dimension
- `zeros`, `ones`: constructs a tensor-stacked size Array, filled by zeros/ones
- `length`: returns the total number of grid points
- `size`: returns the number of grid points in each dimension as a tuple
- `ndims`: returns the dimensionality of the domain
- `clamp`: clamps a D-dim point to the domain
- `show`: pretty print the domain
- `LinRange`: returns a LinRange of the i-th dimension
- `rand`: returns a random point in the domain

## Overloaded Statistics methods
- `median`: return the very middle/center point of the domain


## Examples
```julia
bds = include("BoxDomains.jl")

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

length(sDomain) # get how many grid points in total
size(sDomain)   # get how many grid points in each dimension as a D-int tuple
ndims(sDomain)  # get the dimensionality of the domain

LinRange(sDomain, 1, 50) # creates a LinRange for the 1st dimension, 50 points

CartesianIndices(sDomain) # creates a CartesianIndices for the domain
LinearIndices(sDomain)    # creates a LinearIndices for the domain

map(sum, sDomain) # map a function over the grid points
map!(sum, zeros(sDomain), sDomain) # map a function over the grid points, store

```
"""
struct TensorDomain{D} <: AbstractBoxDomain{D}
    dimnames::NTuple{D,Symbol}
    lb::SV64{D}
    ub::SV64{D}
    Ns::NTuple{D,Int}

    function TensorDomain(
        lb      ::AbsV, 
        ub      ::AbsV, 
        Ns      ::Iterable{Int} ;
        dimnames::Union{Nothing, Iterable} = nothing
    )
        d = length(lb)
        @assert d > 0 "lb and ub must be non-empty"
        @assert length(ub) == d "lb and ub must have the same length"
        @assert all(lb .<= ub) "lb and ub must be consistent"
        @assert all(isfinite, lb) "lb must be finite"
        @assert all(isfinite, ub) "ub must be finite"
        @assert all(Ns .> 0) "Ns must be positive"
        _dnames = if isnothing(dimnames)
            Tuple([ Symbol(:x,i) for i in 1:d ])
        else
            @assert length(dimnames) == d "dimnames size mismatch"
            Tuple([ Symbol(dname) for dname in dimnames ])
        end
        new{d}(
            _dnames,
            SV64{d}(lb),
            SV64{d}(ub),
            Tuple(Ns)
        )
    end # constructor
end # TensorDomain
# ------------------------------------------------------------------------------
function Base.show(io::IO, td::TensorDomain{D}) where D
    println(io, "TensorDomain{$D}")
    for j in 1:D
        _nm = td.dimnames[j]
        _lb = round(td.lb[j], digits=3)
        _ub = round(td.ub[j], digits=3)
        _ns = td.Ns[j]
        
        println(io, "  $(_nm): [$(_lb), $(_ub)], #nodes = $(_ns)")
    end
    println(io, "  #points: $(td.Ns |> prod)")
end
# ------------------------------------------------------------------------------
"""
    TensorDomain(Ns::Iterable{Int})

Default constructor for TensorDomain, with lower bound 0, upper bound 1,which
is the unit hypercube.
"""
function TensorDomain(Ns::Iterable{Int})
    D = length(Ns)
    TensorDomain(zeros(D), ones(D), Ns)
end
# ------------------------------------------------------------------------------
"""
    TensorDomain(bd::BoxDomain{D}, Ns::Iterable{Int})

Construct a `TensorDomain` from a `BoxDomain` by specifying the number of grid 
points in each dimension.
"""
function TensorDomain(bd::BoxDomain{D}, Ns::Iterable{Int}) where D
    TensorDomain(bd.lb, bd.ub, Ns, dimnames = bd.dimnames)
end
# ------------------------------------------------------------------------------
function BoxDomain(td::TensorDomain)
    return BoxDomain(td.lb, td.up, dimnames = td.dimnames)
end






# ------------------------------------------------------------------------------
function Base.getindex(td::TensorDomain{D}, i::Int)::LinRange64 where D
    @assert 1 <= i <= D "index out of range"
    LinRange{F64,I64}(td.lb[i], td.ub[i], td.Ns[i])
end
function Base.getindex(
	td::TensorDomain{D}, 
    i2j::UnitRange{Int}
)::TensorDomain where D
	# return a sub-domain
    return TensorDomain(
    	td.lb[i2j],
        td.ub[i2j],
        td.Ns[i2j],
        dimnames = td.dimnames[i2j]
    )
end
function Base.getindex(
    td::TensorDomain{D},
    i2j::Vector{Int}
)::TensorDomain where D
    # return a sub-domain
    return TensorDomain(
        td.lb[i2j],
        td.ub[i2j],
        td.Ns[i2j],
        dimnames = td.dimnames[i2j]
    )
end
function Base.getindex(
    td   ::TensorDomain{D},
    dname::Symbol
)::LinRange64 where D
    for i in 1:D
        if td.dimnames[i] == dname
            return LinRange{F64,I64}(td.lb[i], td.ub[i], td.Ns[i])
        end
    end
    throw(ArgumentError("dimension name $dname not found"))
end
function Base.getindex(td::TensorDomain{D}, sub::CartesianIndex{D})::V64 where D
    return [td[i][sub[i]] for i in 1:D]
end


# ------------------------------------------------------------------------------
function Base.collect(td::TensorDomain{D})::NTuple{D,LinRange64} where D
    return ntuple(i -> td[i], D)
end

# ------------------------------------------------------------------------------
function Iterators.product(td::TensorDomain{D})::DomainIterator{D} where D
    return Iterators.product([td[i] for i in 1:D]...)
end

# ------------------------------------------------------------------------------
function Base.stack(td::TensorDomain{D})::NM64 where D
    xStack = td |> Iterators.product |> collect |> vec |> stack |> permutedims
    return NamedArray(
        xStack,
        names = (1:size(xStack,1), td.dimnames |> collect),
        dimnames = ("Node", "Dimension")
    )
end

# ------------------------------------------------------------------------------
function Base.Array(td::TensorDomain{D})::Array{F64,D} where D
    return Array{F64,D}(undef, td.Ns...)
end
function Base.zeros(td::TensorDomain{D})::Array{F64,D} where D
    return zeros(F64, td.Ns...)
end
function Base.ones(td::TensorDomain{D})::Array{F64,D} where D
    return ones(F64, td.Ns...)
end

# ------------------------------------------------------------------------------
function Base.length(td::TensorDomain{D})::Int where D
    return prod(td.Ns)
end
function Base.size(td::TensorDomain{D})::NTuple{D,Int} where D
    return Tuple(td.Ns)
end

# ------------------------------------------------------------------------------
function Base.CartesianIndices(td::TensorDomain{D})::CartesianIndices{D} where D
    return CartesianIndices(td.Ns)
end
function Base.LinearIndices(td::TensorDomain{D})::LinearIndices{D} where D
    return LinearIndices(td.Ns)
end


# ------------------------------------------------------------------------------
"""
    map(
        f::Function, 
        td::TensorDomain{D}
    )::Array{F64,D}

Map a function `f(x):R^D -> R` over the grid points of the tensor domain `td`.
Returns a D-dim array of the same size as the array created by `Array(td)`.
The function `f` is supposed to be able to receive a single parameter `x` of
type `NTuple{D,Float64}` and return a single real value that can be converted
to `Float64`.
"""
function Base.map(
    f ::Function,
    td::TensorDomain{D}
)::Array{F64,D} where D
    return map(f, Iterators.product(td))
end
# ------------------------------------------------------------------------------
"""
    map!(
        f::Function, 
        out::Array{F64,D}, 
        td::TensorDomain{D}
    )

Apply a function `f(x):R^D -> R` over the grid points of the tensor domain `td`,
and store the results in the pre-allocated array `out`. The function `f` is
supposed to be able to take a single parameter `x` of type `NTuple{D,Float64}`
and return a single real value that can be converted to `Float64`.
"""
function Base.map!(
    f  ::Function,
    out::Array{F64,D},
    td ::TensorDomain{D}
) where D
    return map!(f, out, Iterators.product(td) |> collect)
end
# ------------------------------------------------------------------------------
"""
    locate(
        x ::AbstractVector, 
        td::TensorDomain{D}
    )::Vector{NTuple{2,Int}} where D

Locates a D-dimensional point `x` in the gridded space `td`. For dimension `d`,
this function finds the two neighbors (along dimension `d`) nodes `td[d][i]` and 
`td[d][j]` such that `td[d][i] <= x[i] <= td[d][j]` brackets the element `x[i]`.
For `x[i]` locates exactly on grid nodes, `j == i`, otherwise `j = i+1`. For `x`
that falls outside the box domain, define the bracket as `(0,1)`
or `(td[d][end],td[d][end])`.

Returns two Int vectors of `i` and `j` indices for each dimension respectively.
"""
function locate(
    x ::AbstractVector, 
    td::TensorDomain{D}
)::Vector{NTuple{2,Int}} where D
    @assert x ∈ td "x is outside the tensor box domain."

    return [
        (
            searchsortedlast(td[d], x[d]),
            searchsortedfirst(td[d], x[d]),
        )
        for d in 1:D
    ]
end
# ------------------------------------------------------------------------------
"""
    neighbors(x::AbstractVector, td::TensorDomain{D}) where D

Find all the nearest on-grid neighbor points for point `x` in the tensor-defined
box domain `td`. The number of the neighbor points ranges from `D` (if all dime-
nsion elements locate exactly on the grid points) to `2^D`.

Returns a vector of `D`-length vectors of integer indices in `td` grids.

## Tips
- to get the values of each neighbor nodes, do `td[res[i] |> CartesianIndex]`,
where `res = bdm.neighbors(x,td)` and `i` is the `i`-th neighbor saved in `res`.
"""
function neighbors(x::AbstractVector, td::TensorDomain{D}) where D
    return Iterators.product(locate(x, td)...) |> unique
end