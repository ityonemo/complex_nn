using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using MLDatasets
using Zygote

@with_kw mutable struct Args
    η::Float32 = 3f-3      # learning rate
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = cpu  # set as gpu, if gpu available
end

function getdata(args)
    # Loading Dataset
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data for flatten the each image into linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return train_data, test_data
end

# define randomization primitives to initialize NN as "complex".
cplx_glorot_uniform(out, in) =
  Flux.glorot_uniform(out, in) .+ Flux.glorot_uniform(out, in) * im
cplx_zeros(args...) = zeros(Complex{Float32}, args...)

# define cplx_dense
cplx_dense(in::Integer, out::Integer, sigma=identity) =
  Dense(in, out, sigma, initW = cplx_glorot_uniform, initb = cplx_zeros)

nonlin(z :: T) where T <: Complex = relu(real(z)) + relu(imag(z)) * im

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
 	    cplx_dense(prod(imgsize), 16, nonlin),
        cplx_dense(16, 16, nonlin),
        cplx_dense(16, nclasses))
end

# redefine crossentropy to be applicable for complex numbers.
function cplx_crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
    return -sum(y .* logsoftmax(real.(ŷ)) .* weight) * 1 // size(y, 2)
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += cplx_crossentropy(model(x), y)
    end
    l/length(dataloader)
end

# redefine "onecold" to work correctly with complex numbers.
import Flux.onecold
onecold(y::AbstractVector{T}) where T <: Complex = onecold(real.(y))

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    #Construct model
    m = build_model()
    train_data = args.device.(train_data)
    test_data = args.device.(train_data)
    m = args.device(m)
    loss(x,y) = cplx_crossentropy(m(x), y)

    # Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)

    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)

    @show accuracy(test_data, m)
end

cd(@__DIR__)
println("beginning training...")
train()
