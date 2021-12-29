@test LearnBase.DataView <: AbstractVector
@test LearnBase.DataView <: LearnBase.AbstractDataIterator
@test LearnBase.DataView{Int} <: AbstractVector{Int}
@test LearnBase.DataView{Int,Vector{Int}} <: LearnBase.AbstractDataIterator{Int,Vector{Int}}
@test LearnBase.AbstractObsView <: LearnBase.DataView
@test LearnBase.AbstractObsView <: LearnBase.AbstractObsIterator
@test LearnBase.AbstractObsView{Int,Vector{Int}} <: LearnBase.DataView{Int,Vector{Int}}
@test LearnBase.AbstractObsView{Int,Vector{Int}} <: LearnBase.AbstractObsIterator{Int,Vector{Int}}
@test LearnBase.AbstractBatchView <: LearnBase.DataView
@test LearnBase.AbstractBatchView <: LearnBase.AbstractBatchIterator
@test LearnBase.AbstractBatchView{Int,Vector{Int}} <: LearnBase.DataView{Int,Vector{Int}}
@test LearnBase.AbstractBatchView{Int,Vector{Int}} <: LearnBase.AbstractBatchIterator{Int,Vector{Int}}

@test LearnBase.DataIterator <: LearnBase.AbstractDataIterator
@test LearnBase.DataIterator{Int,Vector{Int}} <: LearnBase.AbstractDataIterator{Int,Vector{Int}}
@test LearnBase.ObsIterator <: LearnBase.DataIterator
@test LearnBase.ObsIterator <: LearnBase.AbstractObsIterator
@test LearnBase.ObsIterator{Int,Vector{Int}} <: LearnBase.DataIterator{Int,Vector{Int}}
@test LearnBase.ObsIterator{Int,Vector{Int}} <: LearnBase.AbstractObsIterator{Int,Vector{Int}}
@test LearnBase.BatchIterator <: LearnBase.DataIterator
@test LearnBase.BatchIterator <: LearnBase.AbstractBatchIterator
@test LearnBase.BatchIterator{Int,Vector{Int}} <: LearnBase.DataIterator{Int,Vector{Int}}
@test LearnBase.BatchIterator{Int,Vector{Int}} <: LearnBase.AbstractBatchIterator{Int,Vector{Int}}