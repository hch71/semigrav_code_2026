#!/usr/bin/env julia -t 1
# semiclassical gravttational gollapse of a massles scalar field
# generic run script for entropy paper
# © HCH 19.12.2024

# we need Pkg ...
using Pkg
# ... to activate the current directory project environment
Pkg.activate(".")

# include the core module
include("./RadialCollapse.jl")
# use the core module
using .RadialCollapse
# we need JLD2 for file access
using JLD2
# parsing command line arguments
using ArgParse

# desired precision
Re=Float64
# default system size
defR=10.0


### decode command line arguments

# create arg parse object
s=ArgParseSettings()

# populate arg parse object
@add_arg_table s begin
    # Radial extent
    "--rmax", "-R"
    help="radial extent"
    arg_type=Re
    default=defR
    # Number of radial discretization points
    "--Nr","-N"
    help="Number of radial discretization points"
    arg_type=Int
    default=256
    # Number of field components
    "--Nc"
    help="Number of field components (multiplicity)"
    arg_type=Int
    default=0
    # Angular momentum cutoff
    "--lmax"
    help="Angular momentum cutoff"
    arg_type=Int
    default=512
    # PV type
    "--PV"
    help="PV type: 0->no PV  1->polynomial cancelation  >=2->log-poly cancellation"
    arg_type=Int
    default=2
    # PV scale
    "--M","-M"
    help="RMS of PV masses"
    arg_type=Re
    default=12.0
    # Number of iterations
    "--tmax","-T"
    help="Maximal evolution time"
    arg_type=Re
    default=2defR
        # Inner horizon (0 for no horizon)
        "--r0in"
        help="Inner horizon (0 for no horizon)"
        arg_type=Re
        default=0.00
    # Bump type
    "--bump","-b"
    help="Bump type"
    default=nuttall
    # Bump height, i.e. total mass inside the bump
    "--lam"
    help="Bump height (total r_s equivalent)"
    arg_type=Re
    default=3.5
    # Bump position
    "--rofs"
    help="Bump offset from outer boundary"
    arg_type=Re
    default=0.0
    # Bump width
    "--sig","-s"
    help="Bump width"
    arg_type=Re
    default=1.0
    # Out/or inmoving
    "--inout","-i"
    help="Initial relative momentum"
    arg_type=Re
    default=-0.0
    # Time step
    "--dt"
    help="Time step"
    arg_type=Re
    default=1/128
end
# now parse the arguments passed
arg=parse_args(s)
# the PV masses
function pvmc(qc,M)
    # qc ... cancellation type (1=poly)
    if qc==1 # quadratic and quartic cancellations
        # base mass square ratios
        bm2=[0.0,1.0,0.75,0.25]
        # and their multiplicities
        NPV=[1,-1,2,-2]
    else  # other than polynomial cancellations
        if qc==2  # polynomial and log-poly cancellations
            # base mass square ratios
            bm2=[0.0,106.8287106024297,20.17193513394254,3.2370791009314686,84.93712615897233,60.61745998016981]
            # and their multiplicities
            NPV=[1,-1,2,-2,3,-3]
        else # just one PV
            # base mass square ratios
            bm2=[0.0,1.0]
            # and their multiplicities
            NPV=[1,-1]
        end
    end
    # minimal PV mass square
    msm=minimum(bm2[2:end])
    # scaled PV mass squares
    PVM2=bm2/msm*M^2
    # return masses and multiplicities
    return PVM2,NPV
end


# select PV scheme
PVM2,NPV=pvmc(arg["PV"],arg["M"])

# the half time step for evolution 
Δt=arg["dt"]
# radial discretization
Δr=arg["rmax"]/arg["Nr"]
# bump position
bR=arg["rmax"]-arg["rofs"]
# inital bump
world=State{Re}(r=[i*Δr for i=1:arg["Nr"]],shape=arg["bump"],ρ=arg["lam"],σ=arg["sig"],R=bR,κ=arg["inout"],M2=PVM2,pv=NPV,Nc=arg["Nc"],lmax=arg["lmax"])

# and the number of iterations
nit=Int(arg["tmax"]/(2*arg["dt"])+1)

# is there already a corresponding file?
if isfile(world.sfn)
    # yes -> open the file
    jldopen(world.sfn) do file
        # find last saved timeslice
        tin=last(keys(file))
        # load its latest time slice as initial state
        world=file[tin]
        # let the user know that a state was loaded from an existing file
        println("Initial state loaded from existing file - t="*tin)
    end # open file
else # no fiel exists
    # let the user know
    println("No file present - starting from initial state at t=0")
end # file already present
# make the progress visible
flush(stdout)

# call update routine
wx=leapfrog(world,Δt,nit)

# let the user know we are done
println("Run completed at t="*string(wx.t))
