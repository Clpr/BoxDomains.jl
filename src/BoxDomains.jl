module BoxDomains
# ==============================================================================
using StaticArrays, NamedArrays




# ------------------------------------------------------------------------------
# alias
const I64     = Int
const F64     = Float64
const V64     = Vector{Float64}
const M64     = Matrix{Float64}

const SV64{D} = SVector{D,Float64}
const SM64{D} = SMatrix{D,D,Float64}
const SVInt{D}= SVector{D,Int} 

const NV64    = NamedVector{Float64}
const NM64    = NamedMatrix{Float64}

const DictF64 = Dict{Symbol,Float64}

const AbsV    = AbstractVector
const AbsM    = AbstractMatrix

const LinRange64 = LinRange{F64,I64}
const DomainIterator{D} = Iterators.ProductIterator{NTuple{D,LinRange64}}

const Iterable{D} = Union{AbsV{D}, Tuple{Vararg{D}}}




# ------------------------------------------------------------------------------
export AbstractBoxDomain

export centroid, affine, rsg

export BoxDomain
export TensorDomain





# ------------------------------------------------------------------------------
abstract type AbstractBoxDomain{D} <: Any end


#=******************************************************************************
INTERFACE METHODS FOR ALL TYPES OF BOX DOMAINS
******************************************************************************=#
function Base.in(x::AbsV, bd::AbstractBoxDomain{D})::Bool where D
    return all(bd.lb .<= x .<= bd.ub)
end
# ------------------------------------------------------------------------------
function Base.clamp(x::Real, bd::AbstractBoxDomain{D}, i::Int)::F64 where D
    return clamp(x, bd.lb[i], bd.ub[i])
end
# ------------------------------------------------------------------------------
function Base.clamp(x::AbsV, bd::AbstractBoxDomain{D})::AbsV where D
    return clamp.(x, bd.lb, bd.ub)
end
# ------------------------------------------------------------------------------
function Base.ndims(bd::AbstractBoxDomain{D})::Int where D
    return D
end
# ------------------------------------------------------------------------------
function Base.rand(bd::AbstractBoxDomain{D})::V64 where D
    X = rand(F64, D)
    for j in 1:D
        X[j] = X[j] * (bd.ub[j] - bd.lb[j]) + bd.lb[j]
    end
    return X
end
# ------------------------------------------------------------------------------
function Base.rand(bd::AbstractBoxDomain{D}, n::Int)::M64 where D
    X = rand(F64, n, D)
    for (j, xj) in X |> eachcol |> enumerate
        xj .= xj .* (bd.ub[j] - bd.lb[j]) .+ bd.lb[j]
    end
    return X
end
# ------------------------------------------------------------------------------
function Base.LinRange(
    bd::AbstractBoxDomain{D}, 
    i ::Int, 
    n ::Int
)::LinRange64 where D
    return LinRange{F64,I64}(bd.lb[i], bd.ub[i], n)
end
# ------------------------------------------------------------------------------
function Base.LinRange(
    bd   ::AbstractBoxDomain{D}, 
    dname::Symbol,
    n    ::Int
)::LinRange64 where D
    for i in 1:D
        if bd.dimnames[i] == dname
            return LinRange{F64,I64}(bd.lb[i], bd.ub[i], n)
        end
    end
    throw(ArgumentError("dimension name $dname not found"))
end
# ------------------------------------------------------------------------------
"""
    centroid(bd::AbstractBoxDomain{D})::V64 where D

Compute the centroid of the box domain, which is the point in the middle of the
box defined by the lower and upper bounds.
"""
function centroid(bd::AbstractBoxDomain{D})::V64 where D
    return (bd.lb .+ bd.ub) ./ 2
end
# ------------------------------------------------------------------------------
"""
    affine(
        x::AbsV, 
        from_bd::AbstractBoxDomain{D}, 
        to_bd::AbstractBoxDomain{D}
    )::V64 where D

Linearly transform a point x from one box domain to another. The transformation
is defined by the affine mapping between the two box domains. The returned point
is clamped to the bounds of the target box domain.
"""
function affine(
    x      ::AbsV, 
    from_bd::AbstractBoxDomain{D}, 
    to_bd  ::AbstractBoxDomain{D}
)::V64 where D
    Δ = (x .- from_bd.lb) ./ (from_bd.ub .- from_bd.lb)
    return clamp.(
        to_bd.lb .+ (to_bd.ub .- to_bd.lb) .* Δ,
        to_bd.lb,
        to_bd.ub
    )
end
# ------------------------------------------------------------------------------
"""
    affine(
        x::AbsV, 
        from_bd::AbstractBoxDomain{D}, 
        lb::Real = 0.0, 
        ub::Real = 1.0
    )::V64 where D

Linearly transform a point x from one box domain to a equal-sized hypercube with
custom lower and upper bounds. The transformation is useful for normalizing data
to a standard ranges such as [0,1]^D, [-1,1]^D, etc.
"""
function affine(
    x      ::AbsV,
    from_bd::AbstractBoxDomain{D} ;
    lb     ::Real = 0.0,
    ub     ::Real = 1.0,
)::V64 where D
    Δ = (x .- from_bd.lb) ./ (from_bd.ub .- from_bd.lb)    
    return clamp.(lb .+ (ub - lb) .* Δ, lb, ub)
end
# ------------------------------------------------------------------------------
"""
    rsg(
        bd       ::AbstractBoxDomain{D},
        accuracy ::Int ;
        maxlevels::NTuple{D,Int} = ntuple(_ -> accuracy, D)
    )::NM64 where D

Create a 2-based lattice regular sparse grid in D-dimensional space with the
given accuracy. The grid is defined by the box domain `bd` and the maximum level
of refinement `maxlevels` along each dimension.

All by-dimension level combinations `k` are kept such that sum(k) <= accuracy
+ D - 1.

Returns a NamedMatrix of the grid points, where each row is a grid point in the
D-dimensional space.

This function is a convenient method for working with middle/high dimensional
polynomial interpolation, quadrature, etc. For the full functionality, pls check
my another package `AdaptiveSG.jl`.
"""
function rsg(
    bd       ::AbstractBoxDomain{D},
    accuracy ::Int ;
    maxlevels::NTuple{D,Int} = ntuple(_ -> accuracy, D)
)::NM64 where D
    @assert accuracy > 0 "accuracy must be positive"
    @assert all(maxlevels .> 0) "maxlevels must be positive"

    # --------------------------------------------------------------------------
    function power2(x::Int)::Int
        return 2 << (x - 1)
    end
    # --------------------------------------------------------------------------
    function get_all_1d_regular_index(l::Int)::Vector{Int}
        if l > 2
            return collect(1:2:(power2(l-1) - 1))
        elseif l == 2
            return [0,2]
        elseif l == 1
            return [1,]
        else
            throw(ArgumentError("level must be >= 1"))
        end
    end
    # --------------------------------------------------------------------------
    function get_x(l::Int, i::Int)::F64
        if (l > 2) & isodd(i)
            return Float64(i / power2(l - 1))
        elseif (l == 2) & (i == 0)
            return 0.0
        elseif (l == 2) & (i == 2)
            return 1.0
        elseif (l == 1) & (i == 1)
            return 0.5
        else
            throw(ArgumentError("invalid level-index pair ($l, $i)"))
        end
    end


    # --------------------------------------------------------------------------
    # let's start

    iterLvls = Iterators.product(Base.OneTo.(maxlevels)...)
    critVal  = accuracy + D - 1

    # filter feasible levels & collect nodes
    nodes = SV64{D}[]

    for ks in iterLvls
        (sum(ks) > critVal) && continue

        iterIndices = Iterators.product([
            get_all_1d_regular_index(k) for k in ks
        ]...)
        for idx in iterIndices
            x01 = get_x.(ks, idx)  # D-dim point in [0,1]^D
            x   = clamp(
                bd.lb .+ (bd.ub .- bd.lb) .* x01,
                bd.lb,
                bd.ub
            )
            push!(nodes, SV64{D}(x))
        end
    end # ks

    return NamedArray(
        nodes |> stack |> permutedims,
        names = (1:length(nodes), bd.dimnames |> collect),
        dimnames = ("Node", "Dimension")
    )
end
# ------------------------------------------------------------------------------
function chebnodesU(lb::Real, ub::Real, n::Int)::Vector{Float64}
    # Generate ChebyshevU nodes in the interval [lb, ub]
    @assert n >= 2 "n must be at least 2"
    res = -cos.((0:n-1) .* π ./ (n-1)) # the chebU formula generates n+1 nodes
    res .+= 1.0                     # shift to [0,2]
    res .*= (ub - lb) / 2.0          # scale to [0,ub-lb]
    res .+= lb                       # shift to [lb,ub]
    return res
end
# ------------------------------------------------------------------------------
function chebnodesT(lb::Real, ub::Real, n::Int)::Vector{Float64}
    # Generate ChebyshevT nodes in the interval [lb, ub]
    @assert n >= 1 "n must be at least 1"
    res = -cos.(
        ((0:n-1) .+ 0.5) .* π ./ n
    )
    res .+= 1.0                     # shift to [0,2]
    res .*= (ub - lb) / 2.0          # scale to [0,ub-lb]
    res .+= lb                       # shift to [lb,ub]
    return res
end 
















#=******************************************************************************
STANDARD BOX DOMAIN
******************************************************************************=#
"""
    BoxDomain{D} <: AbstractBoxDomain

A box/rectangle domain in D-dimensional space. Defined as a product of intervals
in each dimension.

Supports named access to the intervals, and iteration over the intervals.


## Example
```julia
bds = include("BoxDomains.jl")

# test: shortcut construction, hypercube in D-dim
domain = bds.BoxDomain(3)

# test: standard construction
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

LinRange(domain, 1, 50)  # a LinRange for the 1st dimension, 50 points
LinRange(domain, :k, 50) # indexed by dimension name

bds.centroid(domain)    # the centroid of the domain, (lb + ub) / 2

bds.affine(x, domain, bds.BoxDomain(3)) # affine transform to a hypercube
bds.affine(x, domain, lb = -1, ub = 1) # affine transform to [-1,1]^D

bds.rsg(domain, 4) # create a 2-based lattice regular sparse grid, accuracy 4

```
"""
struct BoxDomain{D} <: AbstractBoxDomain{D}
    dimnames::NTuple{D, Symbol}
    lb::SV64{D}
    ub::SV64{D}

    function BoxDomain(
        lb::Iterable ,
        ub::Iterable ;
        dimnames::Union{Iterable, Nothing} = nothing
    )
        d = length(lb)
        @assert d > 0 "lb and ub must be non-empty"
        @assert length(ub) == d "lb and ub must have the same length"
        @assert all(lb .<= ub) "lb and ub must be consistent"
        @assert all(isfinite, lb) "lb must be finite"
        @assert all(isfinite, ub) "ub must be finite"
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
        )
    end # constructor
end # BoxDomain
# ------------------------------------------------------------------------------
function Base.show(io::IO, domain::BoxDomain{D}) where D
    println(io, "BoxDomain{", D, "}")
    for i in 1:D
        println(
            io, 
            "  ", domain.dimnames[i], 
            " ∈ [", domain.lb[i], ", ", domain.ub[i], "]"
        )
    end
end
# ------------------------------------------------------------------------------
"""
    BoxDomain(D::Int)

Create a default D-dimensional box domain with all intervals set to [0,1] which 
is a hypercube in D-dimensional space.
"""
BoxDomain(D::Int) = BoxDomain(zeros(D), ones(D))





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






# ==============================================================================
end # module