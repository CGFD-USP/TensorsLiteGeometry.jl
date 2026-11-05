using TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra
import TensorsLiteGeometry: integrate
using Test

displacement = (1.3𝐢 - 1.3𝐣)
a = Vec() + displacement
b = 2.0𝐢 + displacement
c = 2.0𝐢 + 2.0𝐣 + displacement
d = 2.0𝐣 + displacement

@testset "Helper Functions" begin
    @test closest(0.9𝐢 + 0.9𝐣, 0.1𝐢 + 0.9𝐣, 1.0, 1.0) ≈ 1.1𝐢 + 0.9𝐣
    @test closest(0.9𝐢 + 0.9𝐣, 0.1𝐢 + 0.1𝐣, 1.0, 1.0) ≈ 1.1𝐢 + 1.1𝐣
    @test closest(0.9𝐢 + 0.9𝐣, 0.9𝐢 + 0.1𝐣, 1.0, 1.0) ≈ 0.9𝐢 + 1.1𝐣
    @test closest(0.1𝐢 + 0.9𝐣, 0.9𝐢 + 0.1𝐣, 1.0, 1.0) ≈ -0.1𝐢 + 1.1𝐣
    @test closest(0.1𝐢 + 0.9𝐣, 0.9𝐢 + 0.9𝐣, 1.0, 1.0) ≈ -0.1𝐢 + 0.9𝐣
    @test closest(0.1𝐢 + 0.1𝐣, 0.9𝐢 + 0.9𝐣, 1.0, 1.0) ≈ -0.1𝐢 - 0.1𝐣
    @test closest(0.1𝐢 + 0.1𝐣, 0.1𝐢 + 0.9𝐣, 1.0, 1.0) ≈ 0.1𝐢 - 0.1𝐣
    @test closest(0.9𝐢 + 0.1𝐣, 0.1𝐢 + 0.9𝐣, 1.0, 1.0) ≈ 1.1𝐢 - 0.1𝐣
    @test closest(0.5𝐢 + 0.5𝐣, 0.7𝐢 + 0.7𝐣, 1.0, 1.0) == 0.7𝐢 + 0.7𝐣

    @test periodic_to_base_point(1.5𝐢+ 1.2𝐣, 1.0, 1.0) ≈ 0.5𝐢 + 0.2𝐣
    @test periodic_to_base_point(-1.5𝐢+ 1.2𝐣, 1.0, 1.0) ≈ 0.5𝐢 + 0.2𝐣
    @test periodic_to_base_point(-1.5𝐢+ -1.2𝐣, 1.0, 1.0) ≈ 0.5𝐢 + 0.8𝐣
    @test periodic_to_base_point(1.5𝐢+ -1.2𝐣, 1.0, 1.0) ≈ 0.5𝐢 + 0.8𝐣
    @test periodic_to_base_point(5*(1.5𝐢+ -1.5𝐣), 1.0, 1.0) ≈ 0.5𝐢 + 0.5𝐣

    @test isapprox_periodic(1.2𝐢 + 1.3𝐣, 21.2𝐢 + 2.3𝐣, 1.0, 1.0)

    #edge case when point is at the boundary
    #periodic_to_base_point(a, 1, 1) will return 0𝐢+0𝐣 while
    #periodic_to_base_point(b, 1, 1) will return 𝐢+𝐣.
    let a = 0.0𝐢 + 0.0𝐣 , b = -1e-19𝐢 + -1e-19𝐣
        @test isapprox_periodic(a, b, 1.0, 1.0, atol = 2e-16)
    end
end

@testset "Circumcenter" begin
    #Test if vertices are really equidistant to circumcenter
    cen = circumcenter(a, b, c)
    @test norm(cen - a) ≈ norm(cen - b) ≈ norm(cen - c)
end

@testset "Area calculation" begin
    @test area(a, b, c) ≈ 2.0
    @test area([a, b, c]) ≈ 2.0
    @test area([a, c, b], [1, 3, 2]) ≈ 2.0
    @test area(a, b, c, d) ≈ 4.0
    @test area([a, b, c, d]) ≈ 4.0
    @test area([c, a, b, d], [3, 1, 2, 4]) ≈ 4.0
    @test area([a, b + (20.0𝐢 - 10.0𝐣), c], [1, 2, 3], 20.0, 10.0) ≈ 2.0
    @test area([a, b, c + (20.0𝐢 - 10.0𝐣), d], [1, 2, 3, 4], 20.0, 10.0) ≈ 4.0
end

@testset "Centroid calculation" begin
    @test centroid([a, b, c, d]) ≈ ((a + b + c + d) / 4)
    @test centroid([a, b, c, d], 1:4) ≈ ((a + b + c + d) / 4)
    @test centroid([a, b, c, d], 1:4, 20.0, 10.0) ≈ ((a + b + c + d) / 4)
    @test mass_centroid(x -> true, a, b, c) ≈ ((a + b + c) / 3)
    @test mass_centroid(x -> true, [a, b, c, d]) ≈ ((a + b + c + d) / 4)
    @test mass_centroid(x -> true, [a, b, c, d], 1:4) ≈ ((a + b + c + d) / 4)
end

@testset "Integration" begin
    @test integrate(x -> (x⋅x)*x, a, b, c) / integrate(x -> (x⋅x), a, b, c) ≈ mass_centroid(x -> x⋅x, a, b, c)
    let f = x -> (3.0x[1]^2 + 2.0x[2]^2 - x[1] + x[2])
        @test integrate(x -> f(x)*x, [a, b, c, d]) / integrate(f, [a, b, c, d]) ≈ mass_centroid(f, [a, b, c, d])
    end
end

@testset "in_triangle function" begin
    @test in_triangle(a, a, b, c)[1]
    @test in_triangle(b, a, b, c)[1]
    @test in_triangle(c, a, b, c)[1]
    @test in_triangle((a + b + c) / 3, a, b, c)[1]
    @test !(in_triangle(3𝐢 + displacement, a, b, c)[1])
end

@testset "in_polygon function" begin
    @test in_polygon(a, [a, b, c, d], [1, 2, 3, 4])
    @test in_polygon(b, [a, b, c, d], [1, 2, 3, 4])
    @test in_polygon(c, [a, b, c, d], [1, 2, 3, 4])
    @test in_polygon(d, [a, b, c, d], [1, 2, 3, 4])
    @test !in_polygon(Vec(), [a, b, c, d], [1, 2, 3, 4])
    @test in_polygon(d, [a, b, c + (20.0𝐢 - 10.0𝐣), d], [1, 2, 3, 4], 20.0, 10.0)
end

@testset "is_obtuse function" begin
    a1 = 0.0𝐢 + 0.0𝐣
    b1 = 1.0𝐢 + 0.0𝐣
    c1 = -0.1𝐢 + 1.0𝐣
    @test is_obtuse(a1, b1, c1)

    a2 = a1
    b2 = b1
    c2 = 1.1𝐢 + 1.0𝐣
    @test is_obtuse(a2, b2, c2)

    a3 = 0.0𝐢 + 1.1𝐣
    b3 = b1
    c3 = 1.0𝐢 + 1.0𝐣
    @test is_obtuse(a3, b3, c3)

    a4 = a1
    b4 = b1
    c4 = 0.5𝐢 + 1.0𝐣
    @test !is_obtuse(a4, b4, c4)
end

@testset "Circle x polygon intersection" begin
    c = 1.2𝐢 - 0.4𝐣
    r = 10.0
    r2 = r * r

    v1 = 5.0𝐢 + 2.0𝐣
    v2 = 20.0𝐢 + 0.0𝐣

    #Test if result is really in the circle
    @test norm(circle_edge_intersection(v1, v2, c, r2) - c) ≈ r
    #Test if result is really in the line (v2-v1) by seeing if (p-v1) × (v2-v1) ≈ 0.0𝐤
    @test isapprox(norm((circle_edge_intersection(v1, v2, c, r2) - v1) × (v2 - v1)), 0.0; atol = 10 * eps())

end

@testset "Circle x polygon intersection area" begin
    disp = (10 * rand()) * 𝐢 + (10 * rand()) * 𝐣
    c = Vec() + disp
    r = 2.0
    r2 = r * r
    v1 = 4.0𝐢 + 4.0𝐣 + disp
    v2 = 0.0𝐢 + 4.0𝐣 + disp
    v3 = 0.0𝐢 + 0.0𝐣 + disp
    v4 = 4.0𝐢 + 0.0𝐣 + disp

    polygon = SmallVector{7}((v1, v2, v3, v4))

    @test polygon_circle_intersection_area(c, r2, polygon) ≈ π * r2 / 4
end

@testset "Spherical Geometry" begin
    R = 2.0
    p1 = R * normalize(Vec(1.0, 1.0, 0.0))
    p2 = R * Vec(1.0, 0.0, 0.0)

    @test arc_length(p1, p2) ≈ π * R / 4
    @test arc_midpoint(p2, R * Vec(0.0, 1.0, 0.0)) ≈ p1

    @test all(spherical_triangle_angles(Vec(0.0, 0.0, R), Vec(R, 0.0, 0.0), Vec(0.0, R, 0.0)) .≈ (π / 2, π / 2, π / 2))
    @test all(spherical_triangle_angles(Vec(0.0, 0.0, R), Vec(R, 0.0, 0.0), R * normalize(Vec(1.0, 1.0, 0.0))) .≈ (π / 4, π / 2, π / 2))

    @test spherical_polygon_area(Vec(0.0, 0.0, R), Vec(R, 0.0, 0.0), Vec(0.0, R, 0.0)) ≈ π * R * R / 2
    @test spherical_polygon_area(Vec(0.0, 0.0, R), Vec(0.0, R, 0.0), Vec(R, 0.0, 0.0)) ≈ -(π * R * R / 2)
    @test spherical_polygon_area(Vec(0.0, 0.0, R), Vec(R, 0.0, 0.0), R * normalize(Vec(1.0, 1.0, 0.0)), Vec(0.0, R, 0.0)) ≈ π * R * R / 2
    @test spherical_polygon_area([Vec(0.0, 0.0, R), Vec(R, 0.0, 0.0), R * normalize(Vec(1.0, 1.0, 0.0)), Vec(0.0, R, 0.0)]) ≈ π * R * R / 2
    @test spherical_polygon_area([Vec(0.0, 0.0, R), R * normalize(Vec(1.0, 1.0, 0.0)), Vec(R, 0.0, 0.0), Vec(0.0, R, 0.0)], [1, 3, 2, 4]) ≈ π * R * R / 2

    lon1 = 20 * π / 180
    lon2 = 20 * π / 180
    lon3 = 25 * π / 180

    lat1 = 45 * π / 180
    lat2 = 35 * π / 180
    lat3 = 40 * π / 180

    v1 = lonlat_to_position(R, lon1, lat1)
    v2 = lonlat_to_position(R, lon2, lat2)
    v3 = lonlat_to_position(R, lon3, lat3)

    @test all(position_to_lonlat(v1) .≈ (lon1, lat1))
    @test all(position_to_lonlat(v2) .≈ (lon2, lat2))
    @test all(position_to_lonlat(v3) .≈ (lon3, lat3))

    @test in_spherical_triangle(v1, v1, v2, v3)
    @test in_spherical_triangle(v2, v1, v2, v3)
    @test in_spherical_triangle(v3, v1, v2, v3)

    v = lonlat_to_position(R, 22 * π / 180, 40 * π / 180)
    @test in_spherical_triangle(v, v1, v2, v3)

    v_out = lonlat_to_position(R, 26 * π / 180, 40 * π / 180)
    @test !in_spherical_triangle(v_out, v1, v2, v3)

    lon4 = 25 * π / 180
    lat4 = 48 * π / 180
    v4 = lonlat_to_position(R, lon4, lat4)

    @test in_spherical_polygon(v1, [v1, v4, v2, v3], [1, 3, 4, 2])
    @test in_spherical_polygon(v2, [v1, v4, v2, v3], [1, 3, 4, 2])
    @test in_spherical_polygon(v3, [v1, v4, v2, v3], [1, 3, 4, 2])
    @test in_spherical_polygon(v4, [v1, v4, v2, v3], [1, 3, 4, 2])

    v = lonlat_to_position(R, 24 * π / 180, 45 * π / 180)
    @test in_spherical_polygon(v, [v1, v4, v2, v3], [1, 3, 4, 2])
    @test !in_spherical_polygon(v_out, [v1, v4, v2, v3], [1, 3, 4, 2])

    @test eastward_vector(Vec(1,0,0)) ≈ 𝐣
    @test eastward_vector(Vec(0,1,0)) ≈ -𝐢
    @test eastward_vector(Vec(-1,0,0)) ≈ -𝐣
    @test eastward_vector(Vec(0,-1,0)) ≈ 𝐢
    @test eastward_vector(Vec(1,1,0)) ≈ normalize(𝐣 - 𝐢)
    @test eastward_vector(Vec(-1,-1,0)) ≈ normalize(𝐢 - 𝐣)

    @test northward_vector(Vec(1,0,0)) ≈ 𝐤
    @test northward_vector(Vec(1,1,0)) ≈ 𝐤
    @test northward_vector(Vec(1,-1,0)) ≈ 𝐤
    @test northward_vector(Vec(1,0,-1)) ≈ normalize(𝐢 + 𝐤)
    @test northward_vector(Vec(1,0,1)) ≈ normalize(-𝐢 + 𝐤)
    p = Vec(rand(), rand(), rand())
    @test eastward_vector(p) × northward_vector(p) ≈ normalize(p)

end
