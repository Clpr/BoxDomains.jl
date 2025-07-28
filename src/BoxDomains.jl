module BoxDomains
# ==============================================================================
using StaticArrays, NamedArrays

import Combinatorics as cmb # for neighbor nodes searching


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
export CustomTensorDomain




# ------------------------------------------------------------------------------
abstract type AbstractBoxDomain{D} <: Any end


# interface/shared methods
include("interface.jl")


# standard box domain representation
include("box.jl")


# tensor/Cartesian product uniform gridded space
include("tensor.jl")


# tensor/Cartesian product custom gridded space
include("custom.jl")












# ==============================================================================
end # module