# module containing evolution types and routines for semiclassical 
# radial collapse
# © HCH 24.1.2023

# development help 
using ExportAll

# for easier generic precision entry
using ChangePrecision
# parameters module for easy structure packing and unpacking
using Parameters
# routines for banded matrices (mosly q and K)
using BandedMatrices
# more intuitive equality for structs
using StructEquality
# We need linear algebra
using LinearAlgebra
# file I/O
using JLD2

# exporting the central types
export State
# export update routine
export leapfrog
# export the bump function
export nuttall
# import algebra for overloading
import Base:*
# export block diagonal algebra
export *
# import conjugate for overloading
import LinearAlgebra:adjoint
# export transpose
export adjoint

# strip a string of its whitespaces
strip(a)=filter(x -> !isspace(x), a)

### print a  message if the  flag is set
function printdb(args...)
    # is debug variable defined?
    if @isdefined debug
        # print the debug message
        !debug || println(args...)
    end
end


### structure of one K-matrix and its eigenmodes
@with_kw struct Kmat{Re<:Real}
    @changeprecision Re begin
        # the K-matrix itself
        mat::BandedMatrix{Re}  
        # the eigenvalues ω
        ω::Vector{Re}
        # the eigenvectors V
        V::Matrix{Re}
    end
end

# define Kmat equality by value
@def_structequal Kmat

### structure containing current state of system
@with_kw struct State{Re<:Real}
    # Re is our default real type
    @changeprecision Re begin
        # current simulation time
        t::Re=0.0
        # innermost radius (where d=0)
        r0::Re=0.0
        # radial coordinates
        r::Vector{Re}=[r0+i*0.1 for i=1:100]
        # number of radial discretization steps
        Nr::Int=size(r,1)
        # discretized derivative forward number of points
        drp::Int=1
        # discretized derivative backward number of points
        drm::Int=0
        # the partial derivative operator as banded matrix
        ∂=dmat(r,drp,drm)
     
        # number of identical field components per field (0 for quenched approximation)
        Nc::Int=0
        # squared masses of the fields
        M2::Vector{Re}=[0.0]
        # number of masses - one more than Pauli-Villars fields
        Nm::Int=size(M2,1)
        # signs and weights of fields (physical field is alvays positive)
        pv::Vector{Int}=[(-1)^i for i=0:Nm-1]

        # maximum angular momentum component considered
        lmax::Int=0
        
        # bump shape
        shape=nuttall
        # bump height
        ρ::Re=3.5
        # bump center
        R::Re=9.0
        # bump half width
        σ::Re=1.0
        # bump direction (-1.0 in, 1.0 out)
        κ::Re=1.0
        # the hamiltonian density
        h::Matrix{Re}=bump(shape,ρ,R,σ,M2[1],κ,∂,r0,r)

        # metric parameter α̂
        α̂::Vector{Re}=radintα̂(h[:,1],r0,r)
        # metric parameter d
        d::Vector{Re}=radintd(h[:,1],h[:,2],r0,r)

        # the current derivative matrix
        q=qmat(∂,α̂,d,r)
        # and the corresponding K matrices
        K=Kmat(q,lmax,M2,α̂,d,r)

        # average for first mass l=0 mode
        χ::Matrix{Re}=iniχ(h[:,1],∂,Binv(K[1,1],α̂,d,r),κ,r)
        # time evolution matrix
        T::Array=idspread(1.0,Nr,lmax,Nm)

        # snapshot time intervals
        dtsn::Re=10.0
        # next snapshot time
        tsnn::Re=0.0
        # snapshot time intervals
        dtobs::Re=0.25
        # next snapshot time
        tobsn::Re=0.0
        # base name of the run
        bnr::String=strip("$(last(r))_$(Nr)_$(Nc)_$(ρ)_$(R)_$(σ)_$(drp)_$(drm)_$(lmax)_$(M2)_$(pv)")
        # name of the snapshot file
        sfn::String=strip("snapshots/world_"*bnr*".jld2")
        # name of the observable file
        ofn::String=strip("obs/obs_"*bnr*".jld2")
    end # @changeprecision Re
end # struct State

# define State equality by value
@def_structequal State

### trivial diagonal correlation array initialization
function idspread(v::Re,Nr::Int,lmax::Int,Nm::Int) where {Re<:Real}
    # v ... diagonal value
    # Nr,lmax,Nm ... additional dimensions

    # init to zeros
    mat::Array{typeof(v)}=zeros(Nr,2,Nr,2,lmax+1,Nm)
    # the desired diagonal matrix
    in=Matrix(v*I,2,2)
    # loop over indices
    for i=1:Nr, li=1:lmax+1, m=1:Nm
        # initialize the matrix
        mat[i,:,i,:,li,m]=in
    end
    return mat
end

### a nuttall bump
function nuttall(R::Re,σ::Re,r::Re) where {Re<:Real}
    # R  ... peak position
    # σ  ... width
    # r  ... radial coordinate

    # invoke generic precision
    @changeprecision Re begin
        # Nuttall coefficients
        a=[-0.487396;0.144232;-0.012604]
    end
    # 0th (constant) coefficient
    a0=-sum(a)
    # normalization factor
    N=σ*(2a0^2+sum(a.^2))
    # argument of the cos term
    arg=π*(r-R+σ)/σ
    # sum the terms and return the result
    return (a0+sum(a[k]*cos(k*arg) for k in 1:3))^2/N
end

### a nuttall bump
function nuttall(R::Re,σ::Re,r0::Re,r::Vector{Re}) where {Re<:Real}
    # R  ... peak position
    # σ  ... width
    # r0 ... innermost radius
    # r  ... radial coordinates
    
    # result vector length
    Nr=length(r)
    # create result vector
    f::Vector{Re}=zeros(Nr)
    # whether a point in inside the bump
    inrange(i)=abs(R-r[i])<σ
    # loop over all points in the range
    for i in filter(inrange,1:Nr)
        # get the function value
        f[i]=nuttall(R,σ,r[i])
        # innermost radius?
        if i==0
            # yes -> multiply with innermost shell thickness
            f[i]*=(r[1]-r0)
        else # no -> multiply with usualshell thickness
            f[i]*=(r[i]-r[i-1])
        end
    end # loop over points

    # return the window function
    return f
end

### definition of the function (exp(x)-1)/x
function expc(x)
    # nontrivial order in the Horner form. 18 is sufficient for BigFloat
    Nhorn=18

    # check if we are in a numerically safe range
    if abs(x)>3//10
        # yes we are, just evaluate integrate
        return (exp(x)-1)/x
    else # too close to x=0, use the Horner expansion
        # construct (x/2)^2 for constructing (sinh(x/2)-1)/x
        xx=(x/2)^2
        # start with trivial result
        res=one(x)
        # loop over powers, starting at the highers
        for i in Nhorn:-1:1
            # build up Horner form from inside out
            res=res*xx/(2*i*(2*i+1))+1
        end
        # res now contains approximation to (sinh(x/2)-1)/x - multiply with e^(x/2) for final result
        return exp(x/2)*res
    end
end

### radial integration of the metric parameter d assuming δ-shells
function radintd(h0::Vector{Re},m::Vector{Re},r0::Re,r::Vector{Re}) where {Re<:Real}
    # h0 ... non-static part of pseudohamiltonian density
    # m  ... static part of pseudohamiltonian density
    # r0 ... innermost radius
    # r  ... radial coordinates

    # lenght of the vectors
    Nr=length(r)
    # check input vectors are of same length
    (Nr==length(h0) && Nr==length(m)) || throw(ArgumentError("Size mismatch in radint!"))
    # initialize result vectors
    d::Vector{Re}=zeros(Nr)
    
    # first point outsde r0 (d0=0)
    d[1]=(r[1]-r0)*exp(-h0[1])+m[1]*expc(-h0[1])
    # loop over radial points, inside out
    for i in 2:Nr
        # update d
        d[i]=(d[i-1]+r[i]-r[i-1])*exp(-h0[i])+m[i]*expc(-h0[i])
    end
    
    # return d
    return d
end

### radial integration of the metric parameter α̂ assuming δ-shells
function radintα̂(h0::Vector{Re},r0::Re,r::Vector{Re}) where {Re<:Real}
    # h0 ... non-static part of pseudohamiltonian density
    # m  ... static part of pseudohamiltonian density
    # r0 ... innermost radius
    # r  ... radial coordinates

    # lenght of the vectors
    Nr=length(r)

    # initialize result vector
    α̂::Vector{Re}=ones(Nr)
    # compute α̂ with BC α̂[Nr]=1
    for i in Nr:-1:2
        # integrate shells from outside in
        α̂[i-1]=α̂[i]*exp(-h0[i])
    end

    # and return the result
    return α̂
end

### radial integration of the metric assuming δ-shells
function radint(h0::Vector{Re},m::Vector{Re},r0::Re,r::Vector{Re}) where {Re<:Real}
    # h0 ... non-static part of pseudohamiltonian density
    # m  ... static part of pseudohamiltonian density
    # r0 ... innermost radius
    # r  ... radial coordinates

    # return the metric parameters
    return radintα̂(h0,m,r0,r),radintd(h0,m,r0,r)
end

### function to optimally approximate the derivative by a finite difference operator
function numder(r::Vector{Re},k::Int) where {Re}
    # r ... array of discretized coordinates
    # k ... point where derivative is constructed

    # find total number of discretization points
    N=length(r)
    # build the array dk_ji=(r_i-r_k)^j/j!
    dk=[(r[i]-r[k])^j/factorial(j) for i=1:N,j=0:N-1]
    #build vector to invert it on
    del=zeros(1,N)
    del[2]=1
    # invert dk on del
    return del/dk
end

### construct the truncated derivative matrix
function dmat(r::Vector{Re},drp::Int,drm::Int) where {Re<:Real}
    # r ...   radial coordinate
    # drp   ... maximum number of forward neighbors for discretized derivative
    # drm   ... maximum number of backward points for discretized derivative
    
    # get size of coordinate vector
    Nr=length(r)
    # allocate the discretized derivative matrix
    dr=zeros(Re,Nr,Nr)
    # exit if derivative does not fit
    drp+drm<Nr || throw(DomainError(drp+drm,"Derivative operator too large"))
    # allocate intermediate coordinate array for discretized derivative
    rd=zeros(Re,drp+drm+1)
    # build the discretized derivative matrix
    for i=1:Nr
        # are backward neighbors outside r range?
        if (i-drm<1)
            # get coordinates that are inside the discretization range
            rd[drm+2-i:drp+drm+1]=r[1:i+drp]
            # loop over points outside discretization range
            for j=drm+1-i:-1:1
                # supplement point with equal coordinate distance
                rd[j]=rd[j+1]-(r[2]-r[1])
            end
            # put truncated derivative discretization into matrix
            dr[i,1:i+drp]=numder(rd,drm+1)[drm+2-i:drp+drm+1]
        # are forward neighbors outside of r range?
        elseif (i+drp>Nr)
            # get coordinates that are inside the discretization range
            rd[1:drm+1+(Nr-i)]=r[i-drm:Nr]
            # loop over points outside discretization range
            for j=drm+2+(Nr-i):drm+1+drp
                # supplement point with equal coordinate distance
                rd[j]=rd[j-1]+(r[Nr]-r[Nr-1])
            end
            # put truncated derivative discretization into matrix
            dr[i,i-drm:Nr]=numder(rd,drm+1)[1:drm+1+(Nr-i)]
        # no neighbors are out of the discretization range
        else
            # get coordinates that are inside the discretization range
            rd=r[i-drm:i+drp]
            # put derivative discretization into matrix
            dr[i,i-drm:i+drp]=numder(rd,drm+1)
        end
    end
    # left multiply with sqrt(α̂*d*r) and right multiply with sqrt(α̂*d/r^3) to obtain q
    return BandedMatrix(dr,(drm,drp))
end

### construct a q-matrix
function qmat(∂::BandedMatrix{Re},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # ∂     ... truncated derivative matrix
    # α̂,d,r ...   metric
    
    return Diagonal(.√(d.*α̂.*r))*∂*Diagonal(.√(d.*α̂./r.^3))
end

### construct a q-matrix OLD
function qmat(α̂::Vector{Re},d::Vector{Re},r::Vector{Re},drp::Int,drm::Int) where {Re<:Real}
    # α̂,d,r ...   metric
    # drp   ... maximum number of forward neighbors for discretized derivative
    # drm   ... maximum number of backward points for discretized derivative
    
    return Diagonal(.√(d.*α̂.*r))*dmat(r,drp,drm)*Diagonal(.√(d.*α̂./r.^3))
end

### construct and diagonalize a K-matrix
function Kmat(q::BandedMatrix{Re},l::Int,M2::Re,α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # q   ... q-matrix that K is based on
    # l   ... angular momentum
    # M2  ... squared mass
    # α̂,d ... metric parameters

    # init K-,Matrix as q^T*q
    K=q'*q+Diagonal(α̂.^2 .*d./r.*(((l+1)*l)./r.^2 .+M2))
    # make an eigenvalue decomposition
    V=eigen(Hermitian(Matrix(K)),sortby=(x->x))
    # and return everything in a Kmat structure
    return Kmat(K,.√V.values,V.vectors)
end

### construct and diagonalize all K-matrices
function Kmat(q::BandedMatrix{Re},lmax::Int,M2::Vector{Re},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # q   ... q-matrix that K is based on
    # l   ... maximum angular momenum
    # M2  ... squared masses
    # α̂,d ... metric parameters
    
    # allocate the result
    K=Matrix{Kmat{Re}}(undef,lmax+1,size(M2,1))
    
    # loop over l and masses
    for l=0:lmax,i=eachindex(M2)
        K[l+1,i]=Kmat(q,l,M2[i],α̂,d,r)
    end # l,i loop

    return K
end


### generating the initial classical bump for massless field
function bump(winf,ρ::Re,R::Re,σ::Re,r0::Re,r::Vector{Re}) where {Re<:Real}
    # winf(R,σ,r0,r) ... window function to be used
    # ρ   ... target total mass
    # R   ... peak position
    # σ   ... width
    # r0  ... innermost radius
    # r   ... radial coordinates
    
    # result vector length
    Nr=length(r)
    # static density
    m::Vector{Re}=zeros(Nr)
    # space for dynamic density
    hc=Vector{Re}(undef,Nr)
    # initial ρ guess
    λ=ρ*σ/20.0
    # initial correction ratio
    cr=1
    # iterate bump height
    for l=1:200
        # compute hc so ϕ̄'=√(λ*winf)
        hc=2*λ*r.*winf(R,σ,r0,r).^2
        # integrate metric parameter d
        d=radintd(hc,m,r0,r)
        # previous correction ratio
        pcr=cr
        # new correction ratio
        cr=ρ/(r[Nr]-d[Nr])
        # converged?
        (1-cr)*(1-pcr)≥0 || break
        # correct ρ by correction ratio
        λ*=cr
    end

    return [hc m]
end

### generating the initial classical bump with arbitrary mass OLD
function bump(winf,ρ::Re,R::Re,σ::Re,M2::Re,κ::Re,∂::BandedMatrix{Re},r0::Re,r::Vector{Re}) where {Re<:Real}
    # winf(R,σ,r0,r) ... window function to be used
    # ρ   ... target total mass
    # R   ... peak position
    # σ   ... width
    # M2  ... squared mass of field
    # κ   ... in/outgoing parameter
    # ∂     ... truncated derivative matrix
    # r0  ... innermost radius
    # r   ... radial coordinates

    # result vector length
    Nr=length(r)
    # is the mass zero?
    if (M2==zero(Re))
        # call the zero mass routine
        return bump(winf,ρ,R,σ,r0,r)
    end
    # desired accuracy for ρ
    ϵ=10*eps(Re)
    # static density
    m::Vector{Re}=zeros(Nr)
    # classical density
    hc::Vector{Re}=zeros(Nr)
    # metric d init
    d=deepcopy(r)
    # metric α̂ init
    α̂::Vector{Re}=ones(Nr) 
    # initial bump height guess
    λ=ρ
    # initial ρ from metric is 0
    ρc=zero(ρ)
    # initial discrepancy
    Δρ=floatmax(ρ)
    # compute m with guessed amplitude λ
    m=λ*winf(R,σ,r0,r)
    # we will need q after the loop
    local q
    # iterate for correct λ
    while true
        # compute q matrix
        q=qmat(∂,α̂,d,r)      
        # update pseudodensity
        hc=(1+κ^2)/(2M2)./α̂.*(q*.√(2r.*m./(α̂.*d))).^2
        # compute estimated total ρ
        ρc=2(sum(hc)+sum(m))
        # has convergence decreased?
        if abs(ρ-ρc)≥5Δρ
            # yes => exit iteration loop
            break
        end
        # update convergence
        Δρ=abs(ρ-ρc)
        # has the iteration converged?
        if Δρ<ϵ
            # yes => exit iteration loop
            break
        end
        # correct λ
        λ*=ρ/ρc
        # recompute m with corrected λ
        m.*=ρ/ρc
        # recompute hc with corrected λ
        hc.*=ρ/ρc
        # obtain metric parameters from radial integration
        α̂,d=radint(hc,m,r0,r)
    end # iteration loop
    # return pseudohamiltonian density functions
    return [hc m]
end

### initialize averages - OLD UNUSED!!!
function iniχ(hc::Vector{Re},m::Vector{Re},α̂::Vector{Re},d::Vector{Re},M2::Re,q::BandedMatrix{Re},K00::Kmat{Re},κ::Re,r::Vector{Re}) where {Re<:Real}
    # hc  ... dynamical part of hamiltonian density
    # m   ... static part of hamiltonian density
    # α̂,d ... metric parameters
    # M2  ... squared mass of field
    # q   ... q matrix
    # K00 ... the l=0 K matrix of the lowest mass
    # κ   ... in/outgoing parameter
    # r   ... radial coordinates

    # result vector length
    Nr=length(r)
    # allocate array of averages
    χ::Matrix{Re}=zeros(Nr,2)
    # unpack the K matrix
    @unpack_Kmat K00
    # the common radical
    cr=.√(2/(1+κ).*α̂.*hc)
    # is the field massless?
    if (M2==0)
        # get the field expectation value from hc
        χ[:,1]=.√ω.*(V'*(q\cr))
    else # massive field
        # get the field expectation value from m
        χ[:,1]=.√ω.*(V'*(.√(2/M2.*r.*m./(α̂.*d))))
    end
    # the momentum expectation value
    χ[:,2]=κ./.√ω.*(V'*cr)
    # return the averages
    return χ
end

### initialize averages, massless case
function iniχ(hc::Vector{Re},∂::BandedMatrix{Re},Bi::Array{Re,3},κ::Re,r::Vector{Re}) where {Re<:Real}
    # hc  ... dynamical part of hamiltonian density
    # ∂   ... radial derivative as a banded matrix
    # Bi  ... Transformation into Williamson basis of l=0 component
    # κ   ... in/outgoing parameter
    # r   ... radial coordinates

    # reconstruct field derivative
    ϕp=.√(hc./r./2) 
    # real space field and momenta
    ψ=[√(2-κ^2)*(∂\ϕp) -κ*ϕp.*r.^2]
    # field/momenta into Williamson form
    return Bi*ψ
end

### time evolution by Δt
function evoscal(T::Array{Re,6},K::Matrix{Kmat{Re}},Δt::Re) where {Re<:Real}
    # T ... matrix that will be time evolved
    # K ... K matrizes and their diagonalizations
    # Δt ... time step
    
    # copy the old transfer matrix
    T̄=Array{Re}(undef,size(T)...)
    # loop over all angular momenta and masses
    for m=1:size(K,2), li=1:size(K,1)
        # construct blocks of the time evolution matrix 
        s=sin(Diagonal(K[li,m].ω*Δt))
        c=cos(Diagonal(K[li,m].ω*Δt))
        # multiply them on T
        T̄[:,1,:,1,li,m]=c*T[:,1,:,1,li,m]+s*T[:,2,:,1,li,m]
        T̄[:,1,:,2,li,m]=c*T[:,1,:,2,li,m]+s*T[:,2,:,2,li,m]
        T̄[:,2,:,1,li,m]=-s*T[:,1,:,1,li,m]+c*T[:,2,:,1,li,m]
        T̄[:,2,:,2,li,m]=-s*T[:,1,:,2,li,m]+c*T[:,2,:,2,li,m]
    end
    # retrun the result
    return T̄
end

### time evolution by Δt, classical part only
function evoscal(T::Array{Re,4},K::Kmat{Re},Δt::Re) where {Re<:Real}
    # T ... matrix that will be time evolved
    # K ... K matrizes and their diagonalizations
    # Δt ... time step
    
    # initialize transfer matrix
    T̄=Array{Re}(undef,size(T)...)

    # construct blocks of the time evolution matrix 
    s=sin(Diagonal(K.ω*Δt))
    c=cos(Diagonal(K.ω*Δt))
    # multiply them on T
    T̄[:,1,:,1]=c*T[:,1,:,1]+s*T[:,2,:,1]
    T̄[:,1,:,2]=c*T[:,1,:,2]+s*T[:,2,:,2]
    T̄[:,2,:,1]=-s*T[:,1,:,1]+c*T[:,2,:,1]
    T̄[:,2,:,2]=-s*T[:,1,:,2]+c*T[:,2,:,2]

    # retrun the result
    return T̄
end

### B matrix (transformation from real space to Williamson form)
function Bmat(K::Matrix{Kmat{Re}},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # K     ... K matrices and their diagonalizations
    # α̂,d,r ... metric
    
    # initialize the result array
    B=Array{Re}(undef,size(r,1),size(r,1),2,size(K)...)

    # loop over all angular momenta and masses
    for mi=1:size(K,2), l=0:size(K,1)-1
        # field component
        B[:,:,1,l+1,mi]=sqrt(Diagonal((α̂.*d)./r.^3))*K[l+1,mi].V/sqrt(Diagonal(K[l+1,mi].ω))
        # momentum component
        B[:,:,2,l+1,mi]=sqrt(Diagonal((α̂.*d)./r.^3))\K[l+1,mi].V*sqrt(Diagonal(K[l+1,mi].ω))
    end

    # give the result
    return B
end

### B matrix (transformation from real space to Williamson form)
function Bmat(K::Kmat{Re},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # K     ... K matrix and its diagonalization
    # α̂,d,r ... metric
    
    # initialize the result array
    B=Array{Re}(undef,size(r,1),size(r,1),2)
    
    # field component
    B[:,:,1]=sqrt(Diagonal((α̂.*d)./r.^3))*K.V/sqrt(Diagonal(K.ω))
    # momentum component
    B[:,:,2]=sqrt(Diagonal((α̂.*d)./r.^3))\K.V*sqrt(Diagonal(K.ω))

    # give the result
    return B
end

### inverse B matrix (transformation from Williamson form to real space)
function Binv(K::Matrix{Kmat{Re}},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # K     ... K matrices and their diagonalizations
    # α̂,d,r ... metric
    
    # initialize the result array
    Bi=Array{Re}(undef,size(r,1),size(r,1),2,size(K)...)

    # loop over all angular momenta and masses
    for mi=1:size(K,2), l=0:size(K,1)-1
        # field component
        Bi[:,:,1,l+1,mi]=sqrt(Diagonal(K[l+1,mi].ω))*K[l+1,mi].V'/sqrt(Diagonal((α̂.*d)./r.^3))
        # momentum component
        Bi[:,:,2,l+1,mi]=sqrt(Diagonal(K[l+1,mi].ω))\K[l+1,mi].V'*sqrt(Diagonal((α̂.*d)./r.^3))
    end
    # give the result
    return Bi
end

### inverse B matrix (transformation from Williamson form to real space)
function Binv(K::Kmat{Re},α̂::Vector{Re},d::Vector{Re},r::Vector{Re}) where {Re<:Real}
    # K     ... K matrix and its diagonalization
    # α̂,d,r ... metric
    
    # initialize the result array
    Bi=Array{Re}(undef,size(r,1),size(r,1),2)
    
    # field component
    Bi[:,:,1]=sqrt(Diagonal(K.ω))*K.V'/sqrt(Diagonal((α̂.*d)./r.^3))
    # momentum component
    Bi[:,:,2]=sqrt(Diagonal(K.ω))\K.V'*sqrt(Diagonal((α̂.*d)./r.^3))

    # give the result
    return Bi
end


### matrix multiplication for block matrices
function *(A::Array{Re,5},B::Array{Re,5}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # loop over blocks, all angular momenta and masses
    for mi=1:size(A,5), l=0:size(A,4)-1, b=1:2
        # block diagonal matrix multiplication
        R[:,:,b,l+1,mi]=A[:,:,b,l+1,mi]*B[:,:,b,l+1,mi]
    end
    # give the result
    return R
end

### matrix multiplication for block on full matrices
function *(A::Array{Re,5},B::Array{Re,6}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks, all angular momenta and masses
    for mi=1:size(A,5), l=0:size(A,4)-1, b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c,l+1,mi]=A[:,:,b,l+1,mi]*B[:,b,:,c,l+1,mi]
    end
    # give the result
    return R
end

### matrix multiplication for block on first mass l=0 component
function *(A::Array{Re,5},B::Array{Re,4}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks, all angular momenta and masses
    for b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c]=A[:,:,b,1,1]*B[:,b,:,c]
    end
    # give the result
    return R
end

### component matrix multiplication for block on full
function *(A::Array{Re,3},B::Array{Re,4}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks, all angular momenta and masses
    for b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c]=A[:,:,b]*B[:,b,:,c]
    end
    # give the result
    return R
end

### component matrix multiplication for block on block
function *(A::Array{Re,3},B::Array{Re,3}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks, all angular momenta and masses
    for b=1:size(A,3)
        # block diagonal matrix multiplication
        R[:,:,b]=A[:,:,b]*B[:,:,b]
    end
    # give the result
    return R
end

### matrix multiplication for full on block matrices
function *(A::Array{Re,6},B::Array{Re,5}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # loop over blocks, all angular momenta and masses
    for mi=1:size(B,5), l=0:size(B,4)-1, b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c,l+1,mi]=A[:,b,:,c,l+1,mi]*B[:,:,c,l+1,mi]
    end
    # give the result
    return R
end

### matrix multiplication for full on full matrices
function *(A::Array{Re,6},B::Array{Re,6}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # loop over blocks, all angular momenta and masses
    for mi=1:size(A,6), l=0:size(A,5)-1, b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c,l+1,mi]=sum(A[:,b,:,s,l+1,mi]*B[:,s,:,c,l+1,mi] for s=1:2)
    end
    # give the result
    return R
end

### block on vector
function *(A::Array{Re,3},B::Matrix{Re}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks
    for b=1:2
        # block diagonal matrix vector multiplication
        R[:,b]=A[:,:,b]*B[:,b]
    end
    
    # give the result
    return R
end

### first mass l=0 full matrix multiplication
function *(A::Array{Re,4},B::Array{Re,6}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # loop over blocks
    for b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c]=sum(A[:,b,:,s]*B[:,s,:,c,1,1] for s=1:2)
    end
    # give the result
    return R
end

### first mass l=0 matrix multiplication
function *(A::Array{Re,4},B::Array{Re,4}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # loop over blocks
    for b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c]=sum(A[:,b,:,s]*B[:,s,:,c] for s=1:2)
    end
    # give the result
    return R
end

### first mass l=0 matrix multiplication for block on full matrices
function *(A::Array{Re,3},B::Array{Re,6}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B[:,:,:,:,1,1])...)
    
    # loop over blocks, all angular momenta and masses
    for b=1:2, c=1:2
        # block diagonal matrix multiplication
        R[:,b,:,c]=A[:,:,b]*B[:,b,:,c,1,1]
    end
    # give the result
    return R
end

### first mass l=0 block matrix on vector
function *(A::Array{Re,5},B::Matrix{Re}) where {Re<:Real}
    
    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over blocks
    for b=1:2
        # block diagonal matrix vector multiplication
        R[:,b]=A[:,:,b,1,1]*B[:,b]
    end
    
    # give the result
    return R
end

### first mass l=0 matrix on vector
function *(A::Array{Re,6},B::Matrix{Re}) where {Re<:Real}
    
    # vector dimension
    N=size(B,1)
    # matrix vector multiplication
    return reshape(reshape(A[:,:,:,:,1,1],2N,2N)*reshape(B,2N),N,2)    
end

### matrix on vector
function *(A::Array{Re,4},B::Matrix{Re}) where {Re<:Real}
    
    # vector dimension
    N=size(B,1)
    # matrix vector multiplication
    return reshape(reshape(A,2N,2N)*reshape(B,2N),N,2)    
end

### apply a banded matrix on all components of a block matrix
function *(A::BandedMatrix{Re},B::Array{Re,3}) where {Re<:Real}

    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over 3rd dimension
    for i=1:size(B,3)
        # apply matrix on each component
        R[:,:,i]=A*B[:,:,i]
    end

    return R
end

### apply a banded matrix on all components of a block matrix
function *(A::BandedMatrix{Re},B::Array{Re,4}) where {Re<:Real}

    # initialize the result array
    R=Array{Float64}(undef,size(B)...)
    
    # loop over 3rd dimension
    for j=1:size(B,4),i=1:size(B,3)
        # apply matrix on each component
        R[:,:,i,j]=A*B[:,:,i,j]
    end

    return R
end

### apply Ω on a matrix
function Ω(A::Array{Re,4}) where {Re<:Real}
   
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)

    # multiply with symplectic Ω
    R[:,1,:,:]=A[:,2,:,:]
    R[:,2,:,:]=-A[:,1,:,:]

    # return the result
    return R
end

### inverse of a symplectic matrix
function sympinv(A::Array{Re,4}) where {Re<:Real}
    # initialize the result array
    R=Array{Float64}(undef,size(A)...)
    
    # do the symplectic inverse
    R[:,1,:,1]=A[:,2,:,2]'
    R[:,2,:,2]=A[:,1,:,1]'
    R[:,2,:,1]=-A[:,2,:,1]'
    R[:,1,:,2]=-A[:,1,:,2]'
    
    # return the result
    return R
end

### find symplectic eigenvalues of C
function symplectic_eigvals(C::Array{Re,4}) where {Re<:Real}

    
    # shape of sympletic matrix as a standard matrix
    sh=(size(C,1)*size(C,2),size(C,3)*size(C,4))
    # compute eigenvalues of iΩC
    λ=eigvals(reshape(im*Ω(C[:,:,:,:,1,1]),sh))
    # number of symplectic eigenmodes
    nse=size(λ,1)÷2
    # allocate symplectic EV array
    μ=Vector{Re}(undef,nse)
    # loop over them
    for i=1:nse
        # and sum opposite eigenmode pairs
        μ[i]=(real(λ[2*nse+1-i])-real(λ[i]))/2
    end
    # return symplectic eigenmodes
    return μ

end

### x*ln(x) with cutoff at 0
function lnt(x::Re) where {Re<:Real}
    x>0 ? x*log(x) : zero(x)
end

### cumulative linear entropy from symplectic eigenmodes
function cle(σ::Vector{Re}) where {Re<:Real}
    # σ ... symplectic eigenvalues

    return sum(1 .-1 ./σ)
end

### von Neumann entropy from symplectic eigenmodes
function vNe(σ::Vector{Re}) where {Re<:Real}
    # σ ... symplectic eigenvalues

    return sum(lnt.((σ.+1)/2)-lnt.((σ.-1)/2))
end

### compute entanglement entropy (von Neumann and linearized) of current state for all r and lmax
function entent(T::Array{Re,6},pv::Vector{Int}) where {Re<:Real}
    # T    ... time evolution matrix in real space
    # pv   ... Pauli-Villars factors
    
    # lmax present in T
    lmax=size(T,5)-1
    # form the covarialce matrix
    C=T*T'
    # size of real space
    Nr=size(T,1)
    # number of masses
    nM=size(T,6)
    # allocate the entropy vectors
    cleio=zeros(Re,Nr,lmax+1,nM)
    vNeio=zeros(Re,Nr,lmax+1,nM)
    vNeios=zeros(Re,Nr,lmax+1,nM)
    pmeio=zeros(Re,Nr,lmax+1,nM)
    # loop over slicings
    for rc=1:Nr
        # is it in the inner half?
        if rc≤Nr÷2
            # take an inner slice of the covariance matrix
            Cs=C[1:rc,:,1:rc,:,:,:]
        else # it is in the outer half
            # take an outer slice of the covariance matrix
            Cs=C[rc+1:end,:,rc+1:end,:,:,:]
        end # inner/outer
        # loop over l modes and masses
        for l=0:lmax,i=1:nM
            # find symplectic eigenvalues
            σ=symplectic_eigvals(Cs[:,:,:,:,l+1,i])
            # find the cumulative linear entropy contribution
            cleio[rc,l+1:lmax+1,i:nM].+=pv[i]*(2l+1)*cle(σ)
            # find the von Neumann entropy contribution
            zvN=vNe(σ)
            # and record it as the per mode entropy
            pmeio[rc,l+1,i]=zvN
            # and add it to the total entropy
            vNeio[rc,l+1:lmax+1,i:nM].+=pv[i]*(2l+1)*zvN
            # as well as to the l-contribution
            vNeios[rc,l+1,i:nM].+=pv[i]*zvN
        end # l,i loop
        # debug-output
        printdb("rc:",rc)
    end # inner rc loop
    
    # return both cumulative entropies, the per mode entropy and per mass entropy
    return vNeio,cleio,vNeios,pmeio
end

### adjoint i.e. transpose of full matrices
function adjoint(A::Array{Re,6}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(3,4,1,2,5,6))
end

### adjoint i.e. transpose of block matrices
function adjoint(A::Array{Re,5}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(2,1,3,4,5))
end

### fast diagonal elements of T*T'
function diagss(T::Matrix{Re}) where {Re<:Real}
    
    # dimensions of the matrix
    n,m=size(T)
    # the result array
    r=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m
        # sum over element squares
        r[j]+=T[j,i]^2
    end # j,i loop
end

### fast diagonal elements of T*T', including proper PV subtraction
function diagssPV(T::Array{Re,4},pv::Vector{Int}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*T[j,i,h,k]^2
    end # j,i loop

    return R
end

### fast diagonal elements of T*T', including proper PV subtraction and static l mode treatment
function diagssPV(T::Array{Re,4},pv::Vector{Int},M2::Vector{Re},l::Int,r::Vector{Re}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields
    # M2 ... squred masses
    # l  ... angular momentum mode
    # r  ... radial coordinate

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*(l*(l+1)+M2[k]*r[j]^2)*T[j,i,h,k]^2
    end # j,i loop

    return R
end

### adjoint i.e. transpose of full matrices
function adjoint(A::Array{Re,6}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(3,4,1,2,5,6))
end

### adjoint i.e. transpose of block matrices
function adjoint(A::Array{Re,5}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(2,1,3,4,5))
end

### fast diagonal elements of T*T'
function diagss(T::Matrix{Re}) where {Re<:Real}
    
    # dimensions of the matrix
    n,m=size(T)
    # the result array
    r=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m
        # sum over element squares
        r[j]+=T[j,i]^2
    end # j,i loop
end

### fast diagonal elements of T*T', including proper PV subtraction
function diagssPV(T::Array{Re,4},pv::Vector{Int}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*T[j,i,h,k]^2
    end # j,i loop

    return R
end

### fast diagonal elements of T*T', including proper PV subtraction and static l mode treatment
function diagssPV(T::Array{Re,4},pv::Vector{Int},M2::Vector{Re},l::Int,r::Vector{Re}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields
    # M2 ... squred masses
    # l  ... angular momentum mode
    # r  ... radial coordinate

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*(l*(l+1)+M2[k]*r[j]^2)*T[j,i,h,k]^2
    end # j,i loop

    return R
end



### adjoint i.e. transpose of full matrices
function adjoint(A::Array{Re,6}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(3,4,1,2,5,6))
end

### adjoint i.e. transpose of block matrices
function adjoint(A::Array{Re,5}) where {Re<:Real}
    # permute the first two pairs od indices
    permutedims(A,(2,1,3,4,5))
end

### fast diagonal elements of T*T'
function diagss(T::Matrix{Re}) where {Re<:Real}
    
    # dimensions of the matrix
    n,m=size(T)
    # the result array
    r=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m
        # sum over element squares
        r[j]+=T[j,i]^2
    end # j,i loop
end

### fast diagonal elements of T*T', including proper PV subtraction
function diagssPV(T::Array{Re,4},pv::Vector{Int}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*T[j,i,h,k]^2
    end # j,i loop

    return R
end

### fast diagonal elements of T*T', including proper PV subtraction and static l mode treatment
function diagssPV(T::Array{Re,4},pv::Vector{Int},M2::Vector{Re},l::Int,r::Vector{Re}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # pv ... signs and weights of fields
    # M2 ... squred masses
    # l  ... angular momentum mode
    # r  ... radial coordinate

    # dimensions of the matrix
    n,m,s,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, h=1:s, k=1:p
        # sum over element squares
        R[j]+=pv[k]*(l*(l+1)+M2[k]*r[j]^2)*T[j,i,h,k]^2
    end # j,i loop

    return R
end

### fast diagonal elements of T*T', including proper PV and vacuum subtraction
function diagssPV(T::Array{Re,4},T0::Array{Re,3},pv::Vector{Int}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # T0 ... and their vacuum subtractions
    # pv ... signs and weights of fields

    # dimensions of the matrix
    n,m,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, k=1:p
        # sum over element squares
        R[j]+=pv[k]*(sum(T[j,i,:,k].^2)-T0[j,i,k]^2)
    end # j,i loop

    return R
end

### fast diagonal elements of T*T', including proper PV subtraction and static l mode treatment
function diagssPV(T::Array{Re,4},T0::Array{Re,3},pv::Vector{Int},M2::Vector{Re},l::Int,r::Vector{Re}) where {Re<:Real}
    # T  ... matrices to be sum-squared
    # T0 ... and their vacuum subtractions
    # pv ... signs and weights of fields
    # M2 ... squred masses
    # l  ... angular momentum mode
    # r  ... radial coordinate

    # dimensions of the matrix
    n,m,p=size(T)
    # the result array
    R=zeros(n)
    
    # loop over elements
    for j=1:n, i=1:m, k=1:p
        # sum over element squares
        R[j]+=pv[p]*(l*(l+1)+M2[k]*r[j]^2)*(sum(T[j,i,:,k].^2)-T0[j,i,k]^2)
    end # j,i loop

    return R
end

### classical energy densities as function of real space averages
function hdens(ψ::Matrix{Re},∂::BandedMatrix{Re},M2::Re,r::Vector{Re}) where {Re<:Real}
    # ψ   ... real space averages
    # ∂   ... radial derivative as a banded matrix
    # M2  ... Squared mass of first field
    # r   ... radial coordinate

    # compute classical density, h0 part and M part
    return [(ψ[:,2].^2 ./r.^3+(∂*ψ[:,1]).^2 .*r) M2*(r.^2).*ψ[:,1].^2]
end

### non point-split energy densities
function hdens(ψ::Matrix{Re},T::Array{Re},∂::BandedMatrix{Re},M2::Vector{Re},pv::Vector{Int},Nc,r::Vector{Re}) where {Re<:Real}
    # ψ   ... real space averages
    # T   ... time evolution matrix
    # ∂   ... radial derivative as a banded matrix
    # M2  ... Squared mass of fields
    # Nc  ... number of field components (Nc=0 is quenched)
    # pv ... signs and weights of fields
    # r   ... radial coordinate
 

    # compute classical densities
    hc=hdens(ψ,∂,M2[1],r)

    # that's all for the quenched case
    Nc>0 || return hc

    # init vacuum contributions to h
    h0Π=zeros(Re,size(r,1))
    h0ϕ=zeros(Re,size(r,1))
    m=zeros(Re,size(r,1))
    # maximum angular momentum
    lmax=size(T,5)-1
    # loop over all angular momentum components
    for l=0:lmax
            # CΠΠ
            h0Π+=(2l+1)/2*diagssPV(T[:,2,:,:,l+1,:],pv)
            # ∂Cϕϕ∂'
            h0ϕ+=(2l+1)/2*diagssPV(∂*T[:,1,:,:,l+1,:],pv)
            # (l(l+1)+r^2M)Cϕϕ
            m+=(2l+1)/2*diagssPV(T[:,1,:,:,l+1,:],pv,M2,l,r)
    end # loop over l

    # and return the end result
    return hc+Nc*[(h0Π./r.^3+h0ϕ.*r) m]
end

### non point-split energy densities with vacuum subtraction
function hsub(B::Array{Re},χ::Matrix{Re},T̂::Array{Re},∂::BandedMatrix{Re},M2::Vector{Re},pv::Vector{Int},Nc,r::Vector{Re}) where {Re<:Real}
    # B   ... transformation into real space
    # ψ   ... averages in Williamson basis
    # T̂   ... time evolution matrix in Williamson basis
    # ∂   ... radial derivative as a banded matrix
    # M2  ... Squared mass of fields
    # Nc  ... number of field components (Nc=0 is quenched)
    # pv ... signs and weights of fields
    # r   ... radial coordinate
 

    # compute classical densities
    hc=hdens(B*(T̂*χ),∂,M2[1],r)

    # that's all for the quenched case
    Nc>0 || return hc
    
    # Transform time evolution matrix into real space
    T=B*T̂
    # init vacuum contributions to h
    h0Π=zeros(Re,size(r,1))
    h0ϕ=zeros(Re,size(r,1))
    m=zeros(Re,size(r,1))
    # maximum angular momentum
    lmax=size(T,5)-1
    # loop over all angular momentum components
    for l=0:lmax
        # CΠΠ
        h0Π+=(2l+1)/2*diagssPV(T[:,2,:,:,l+1,:],B[:,:,2,l+1,:],pv)
        # ∂Cϕϕ∂'
        h0ϕ+=(2l+1)/2*diagssPV(∂*T[:,1,:,:,l+1,:],∂*B[:,:,1,l+1,:],pv)
        # (l(l+1)+r^2M)Cϕϕ
        m+=(2l+1)/2*diagssPV(T[:,1,:,:,l+1,:],B[:,:,1,l+1,:],pv,M2,l,r)
    end # loop over l

    # and return the end result
    return hc+Nc*[(h0Π./r.^3+h0ϕ.*r) m]
end

### non point-split energy densities with and without vacuum subtraction, all intermediate l mode sums
function hall(B::Array{Re},χ::Matrix{Re},T̂::Array{Re},∂::BandedMatrix{Re},M2::Vector{Re},pv::Vector{Int},Nc,r::Vector{Re}) where {Re<:Real}
    # B   ... transformation into real space
    # ψ   ... averages in Williamson basis
    # T̂   ... time evolution matrix in Williamson basis
    # ∂   ... radial derivative as a banded matrix
    # M2  ... Squared mass of fields
    # Nc  ... number of field components (Nc=0 is quenched)
    # pv ... signs and weights of fields
    # r   ... radial coordinate
    
    # number of radial discretization steps
    Nr=size(r,1)
    # maximum angular momentum
    lmax=size(T̂,5)-1

    # allocate the results array with subtraction
    hl=zeros(Re,Nr,lmax+2)
    ml=zeros(Re,Nr,lmax+2)
    # allocate the results array without subtraction
    hn=zeros(Re,Nr,lmax+2)
    mn=zeros(Re,Nr,lmax+2)

    # compute classical densities
    hc=hdens(B*(T̂*χ),∂,M2[1],r)

    # and assign them to the result arrays
    hl[:,1]=hc[:,1]
    ml[:,1]=hc[:,2]
    hn[:,1]=hc[:,1]
    mn[:,1]=hc[:,2]

    # that's all for the quenched case
    Nc>0 || return hc
    
    # Transform time evolution matrix into real space
    T=B*T̂
    # loop over all angular momentum components
    for l=0:lmax
        # CΠΠ
        h0Π=(2l+1)/2*diagssPV(T[:,2,:,:,l+1,:],B[:,:,2,l+1,:],pv)
        # ∂Cϕϕ∂'
        h0ϕ=(2l+1)/2*diagssPV(∂*T[:,1,:,:,l+1,:],∂*B[:,:,1,l+1,:],pv)
        # add up the h contributions
        hl[:,l+2]=hl[:,l+1]+Nc*(h0Π./r.^3+h0ϕ.*r)
        # (l(l+1)+r^2M)Cϕϕ
        ml[:,l+2]=ml[:,l+1]+Nc*(2l+1)/2*diagssPV(T[:,1,:,:,l+1,:],B[:,:,1,l+1,:],pv,M2,l,r)
        # CΠΠ unsubtracted
        h0Π=(2l+1)/2*diagssPV(T[:,2,:,:,l+1,:],pv)
        # ∂Cϕϕ∂' unsubtracted
        h0ϕ=(2l+1)/2*diagssPV(∂*T[:,1,:,:,l+1,:],pv)
        # add up the h contributions
        hn[:,l+2]=hn[:,l+1]+Nc*(h0Π./r.^3+h0ϕ.*r)
        # (l(l+1)+r^2M)Cϕϕ unsubtracted
        mn[:,l+2]=mn[:,l+1]+Nc*(2l+1)/2*diagssPV(T[:,1,:,:,l+1,:],pv,M2,l,r)
    end # loop over l

    # and return the end result
    return hl,ml,hn,mn
end

### main time evolution loop
function leapfrog(world::State{Re},Δt::Re,nit::Int;silent::Bool=true,noobs::Bool=false) where {Re<:Real}
    # world  ... the state to be evolved
    # Δt     ... (half) time step
    # nit    ... number of full time steps to be conducted
    # silent ... no snapshots
    # noobs  ... no observables written

    # unpack the input state
    @unpack_State world
    # set snapshot time to current time
    tsnn=t
    # set observable time to current time
    tobsn=t

    # main loop
    for it=1:nit+1

        # transformation matrix into real space
        B=Bmat(K,α̂,d,r)
        
        ### do measurements
        
        #display(plot(1 .-d./r))
        printdb("t=",t," Birkhoff:",last(r)-last(d))
        
        # is a snapshot due?
        if !silent&(t≥tsnn)
            # pack a state with the current time stamp
            state=@pack_State
            # create the directory if it does not yet exist
            mkpath(dirname(sfn))
            # open the snapshot file in append mode
            jldopen(sfn,"a+") do file
                # check if the time slice is already there
                if haskey(file,string(t))
                    # let the user know
                    println("Time $(t) alredy present in snapshot file!")
                else # it's not there, write it
                    # write the state to the correct time
                    file["$(t)"]=state
                end # time already in file check
            end
            tsnn+=dtsn
        end # snapshot
        # is a measurement due?
        if !noobs&(t≥tobsn)
            mkpath(dirname(ofn))
            # open the entropy observable file in append mode
            jldopen(ofn,"a+") do file
                # check if the time slice is already there
                if haskey(file,string(t))
                    # let the user know
                    println("Time $(t) alredy present in observables file!")
                else # it's not there, write it
                    # let the user know
                    println("Adding time $(t) to observables file.")
                    # compute cummulative linearized and von Neumann entropies
                    vNe,cle,vNes,pme=entent(B*T,pv)
                    # write the von Neumann entropy
                    file["$(t)/vNe"]=vNe
                    # write the single mode von Neumann entropy
                    file["$(t)/vNes"]=vNes
                    # write the per mass single mode von Neumann entropy
                    file["$(t)/pme"]=pme
                    # write the cummulative linearized entropy
                    file["$(t)/cle"]=cle
                    # and the corresponding instantaneous vacuum entropies
                    vNe,cle,vNes,pme=entent(B*idspread(1.0,Nr,lmax,Nm),pv)
                    # write the von Neumann entropy
                    file["$(t)/vNie"]=vNe
                    # write the single mode von Neumann entropy
                    file["$(t)/vNies"]=vNes
                    # write the per mass single mode von Neumann entropy
                    file["$(t)/pmie"]=pme
                    # write the cummulative linearized entropy
                    file["$(t)/clie"]=cle
                    # obtain the energy densities
                    hl,ml,hn,mn=hall(B,χ,T,∂,M2,pv,1,r)
                    # write the subtracted densities
                    file["$(t)/hsub"]=hl
                    file["$(t)/msub"]=ml
                    # write the unsubtracted densities
                    file["$(t)/hunsub"]=hn
                    file["$(t)/munsub"]=mn
                    # write the radii
                    file["$(t)/r"]=r
                    # write the metric parameters
                    file["$(t)/alhat"]=α̂
                    file["$(t)/d"]=d
                end # time already in file check
            end
            # update next measurement time
            tobsn+=dtobs
        end # measurement

        
        # exit main loop after last round of measurements
        it≠nit+1 || break
        
        # explicit time evolution by Δt
        T=evoscal(T,K,Δt)
        
        # cast time evolution matrix into real space
        T=B*T
        
        # auxiliary variable for convergence tracking
        odif=floatmax(Re)
        
        # quenched case?
        if (Nc==0)
            # only first mass l=0 component needed for evolution
            Ke=K[1,1]
            # only l=0 mode
            le=0
            # only first mass
            M2e=M2[1]
        else # unquenched
            # full matrix needed for evolution
            Ke=K
            # all l modes
            le=lmax
            # and all masses
            M2e=M2
        end # quenched/unquenched
        
        # implicit Euler iteration loop
        for ii=1:10
            
            
            # transformation into Williamson form with current metric guess
            Bi=Binv(Ke,α̂,d,r)
            # do the implicit step with current metric guess
            T̄=evoscal(Bi*T,Ke,Δt)
            
            # compute energy densities
            h=hsub(B,χ,T̄,∂,M2,pv,Nc,r)
            # copy old metric
            α̂̄,d̄=α̂,d
            # integrate metric parameter α̂
            α̂=radintα̂(h[:,1],r0,r)
            # integrate metric parameter d
            d=radintd(h[:,1],h[:,2],r0,r)
            # update q matrix
            q=qmat(∂,α̂,d,r)
            # new residue
            ndif=maximum(abs.([α̂̄-α̂ d̄-d]))
            # has the metric converged?
            if (odif≤ndif)
                # exit loop
                break
            end
            # and l=0 K-matrix for first mass
            Ke=Kmat(q,le,M2e,α̂,d,r)          
            # transformation matrix into real space
            B=Bmat(Ke,α̂,d,r)
            
            # update residue
            odif=ndif
            
        end # implicit Euler iteration
        
        # recompute full K matrix
        K=Kmat(q,lmax,M2,α̂,d,r)
        # transformation into Williamson form with iterated metric
        Bi=Binv(K,α̂,d,r)
        # do the implicit step with iterated metric
        T=evoscal(Bi*T,K,Δt)
        # update simulation time
        t+=2Δt
    end # main loop

    # finally pack the state again and return it
    return @pack_State
end

### total mass of a state
function mtot(s::State{Re})  where {Re<:Real}
    return s.r[s.Nr]-s.d[s.Nr]
end
