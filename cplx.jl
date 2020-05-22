using Flux:onecold
using Test

# redefine "onecold" to work correctly with complex numbers.
import Flux.onecold
onecold(y::AbstractVector{T}) where T <: Complex = onecold(real.(y))

@testset "onecold" begin
  a = [1, 2, 5, 3.]
  @test onecold(a) == 3
  a_cplx = [1.0 + 2.0im, 2.0 + 1.0im, -2.0 + 1.0im, -3.0 + 0.0im]
  @test onecold(a_cplx) == 2

  A = [1 20 5
       2 7 6
       3 9 10
       2 1 14]
  @test onecold(A) == [3, 1, 4]

  A_cplx = [1 + 0.3im 20 + 0.15im 5
            2 + 7im   7  + 30im   6
            3 + -10im 9  + -10im  10
            2 + 0im   1  + 5im    14]

  @test onecold(A_cplx) == [3, 1, 4]
end
