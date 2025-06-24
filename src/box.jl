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
bds = include("src/BoxDomains.jl")

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

domain[1]           # get a 2-tuple of (lb,ub) of the 1st dimension
domain[:k]

domain[1:2]         # get a sub-domain of the 1st and 2nd dimensions
domain[[1,2]]
domain[[:k,:l]]

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
# ------------------------------------------------------------------------------
function Base.getindex(domain::BoxDomain{D}, i::Int)::BoxDomain where D
    return BoxDomain(
        [domain.lb[i],], 
        [domain.ub[i],],
        dimnames = (domain.dimnames[i],)
    )
end
function Base.getindex(
	domain::BoxDomain{D}, 
    i2j::UnitRange{Int}
)::BoxDomain where D
	# return a sub-domain
    return BoxDomain(
    	domain.lb[i2j],
        domain.ub[i2j],
        dimnames = domain.dimnames[i2j]
    )
end
function Base.getindex(
    domain::BoxDomain{D},
    i2j::Vector{Int}
)::BoxDomain where D
    # return a sub-domain
    return BoxDomain(
        domain.lb[i2j],
        domain.ub[i2j],
        dimnames = domain.dimnames[i2j]
    )
end
function Base.getindex(
    domain   ::BoxDomain{D},
    dname::Symbol
)::BoxDomain where D
    name2idx = Dict(domain.dimnames .=> (1:length(domain.dimnames)) )
    return BoxDomain(
        [domain.lb[name2idx[dname]],],
        [domain.ub[name2idx[dname]],],
        dimnames = (dname,)
    )
end
function Base.getindex(
    domain::BoxDomain{D},
    dnames::Vector{Symbol}
)::BoxDomain where D

    name2idx = Dict(domain.dimnames .=> 1:length(domain.dimnames))    

    lb     = F64[]
    ub     = F64[]
    dNames = Symbol[]

    for d in dnames
        push!(lb, domain.lb[name2idx[d]])
        push!(ub, domain.ub[name2idx[d]])
        push!(dNames, d)
    end
    return BoxDomain(lb, ub, dimnames = dNames)
end
