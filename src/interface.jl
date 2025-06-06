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
