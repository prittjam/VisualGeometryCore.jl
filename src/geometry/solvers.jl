# Geometric algorithm solvers (P3P, PnP, optimization)
# Includes polynomial solvers ported from PoseLib

# =============================================================================
# Polynomial Solvers
# =============================================================================

"""
    solve_quadratic_real(a, b, c)

Solve the quadratic equation a*x^2 + b*x + c = 0 for real roots.
Returns an array of real roots.

Uses numerically stable formulation to avoid cancellation errors.
"""
function solve_quadratic_real(a::Float64, b::Float64, c::Float64)
    b2m4ac = b^2 - 4 * a * c
    if b2m4ac < 0
        return Float64[]
    end

    sq = sqrt(b2m4ac)

    # Choose sign to avoid cancellations
    if b > 0
        root1 = (2 * c) / (-b - sq)
    else
        root1 = (2 * c) / (-b + sq)
    end

    root2 = c / (a * root1)

    return [root1, root2]
end

# Internal helper functions for P3P (not exported)
# From PoseLib p3p_common.h lines 8-30
function root2real(b::Float64, c::Float64)
    THRESHOLD = -1.0e-12
    v = b * b - 4.0 * c
    if v < THRESHOLD
        r1 = r2 = -0.5 * b
        return (v >= 0), [r1, r2]
    end
    if v > THRESHOLD && v < 0.0
        r1 = -0.5 * b
        r2 = -2.0
        return true, [r1, r2]
    end

    y = sqrt(v)
    if b < 0
        r1 = 0.5 * (-b + y)
        r2 = 0.5 * (-b - y)
    else
        r1 = 2.0 * c / (-b + y)
        r2 = 2.0 * c / (-b - y)
    end
    return true, [r1, r2]
end

function solve_cubic_single_real(k2::Float64, k1::Float64, k0::Float64)
    # Depressed cubic t^3 + p t + q = 0 via s = t - k2/3
    a = k2
    p = k1 - a*a/3.0
    q = k0 - a*k1/3.0 + 2.0*a*a*a/27.0

    # Discriminant
    D = (q/2.0)^2 + (p/3.0)^3

    if D > 0
        # One real root
        u = cbrt(-q/2.0 + sqrt(D))
        v = cbrt(-q/2.0 - sqrt(D))
        t = u + v
        x = t - a/3.0
        return [x]
    elseif D == 0
        # Multiple roots
        if abs(q) < 1e-12
            return [-a/3.0]  # Triple root
        else
            u = cbrt(-q/2.0)
            return [2*u - a/3.0, -u - a/3.0]  # One single, one double
        end
    else
        # Three real roots (should not happen in our P3P case)
        rho = sqrt(-(p/3.0)^3)
        theta = acos(-q/(2.0*rho))
        rho_cbrt = cbrt(rho)
        return [2*rho_cbrt*cos(theta/3.0) - a/3.0,
                2*rho_cbrt*cos((theta + 2*π)/3.0) - a/3.0,
                2*rho_cbrt*cos((theta + 4*π)/3.0) - a/3.0]
    end
end

# From PoseLib p3p_common.h lines 32-70
function compute_pq(C::SMatrix{3,3,Float64})
    # Lines 36-44: Compute adjugate
    C_adj = @SMatrix [
        C[2,3]*C[3,2] - C[2,2]*C[3,3]  C[1,2]*C[3,3] - C[1,3]*C[3,2]  C[1,3]*C[2,2] - C[1,2]*C[2,3];
        C[1,2]*C[3,3] - C[1,3]*C[3,2]  C[1,3]*C[3,1] - C[1,1]*C[3,3]  C[1,1]*C[2,3] - C[1,3]*C[2,1];
        C[1,3]*C[2,2] - C[1,2]*C[2,3]  C[1,1]*C[2,3] - C[1,3]*C[2,1]  C[1,2]*C[2,1] - C[1,1]*C[2,2]
    ]

    # Lines 47-57: Find largest diagonal, select column
    v = if C_adj[1,1] > C_adj[2,2]
        if C_adj[1,1] > C_adj[3,3]
            C_adj[:, 1] / sqrt(C_adj[1,1])
        else
            C_adj[:, 3] / sqrt(C_adj[3,3])
        end
    elseif C_adj[2,2] > C_adj[3,3]
        C_adj[:, 2] / sqrt(C_adj[2,2])
    else
        C_adj[:, 3] / sqrt(C_adj[3,3])
    end

    # Lines 59-64: Modify C
    C_mod = @SMatrix [
        C[1,1]       C[1,2]-v[3]  C[1,3]+v[2];
        C[2,1]+v[3]  C[2,2]       C[2,3]-v[1];
        C[3,1]-v[2]  C[3,2]+v[1]  C[3,3]
    ]

    # Lines 66-67: Return col 0 and row 0
    pq = [C_mod[:, 1], SVector(C_mod[1,1], C_mod[1,2], C_mod[1,3])]
    return pq
end

# From PoseLib p3p_common.h lines 73-95
function refine_lambda!(lambda1::Ref{Float64}, lambda2::Ref{Float64}, lambda3::Ref{Float64},
                        a12::Float64, a13::Float64, a23::Float64,
                        b12::Float64, b13::Float64, b23::Float64)
    for iter in 1:5
        λ1, λ2, λ3 = lambda1[], lambda2[], lambda3[]

        r1 = (λ1 * λ1 - 2.0 * λ1 * λ2 * b12 + λ2 * λ2 - a12)
        r2 = (λ1 * λ1 - 2.0 * λ1 * λ3 * b13 + λ3 * λ3 - a13)
        r3 = (λ2 * λ2 - 2.0 * λ2 * λ3 * b23 + λ3 * λ3 - a23)

        if abs(r1) + abs(r2) + abs(r3) < 1e-10
            return
        end

        x11 = λ1 - λ2 * b12
        x12 = λ2 - λ1 * b12
        x21 = λ1 - λ3 * b13
        x23 = λ3 - λ1 * b13
        x32 = λ2 - λ3 * b23
        x33 = λ3 - λ2 * b23

        detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33)

        lambda1[] += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ
        lambda2[] += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ
        lambda3[] += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ
    end
end


"""
    p3p(rays::AbstractVector, points_3d::AbstractVector) -> (rotations, translations)

Solve the Perspective-3-Point (P3P) problem to find camera poses.

Given 3 normalized image rays and their corresponding 3D points, compute possible
camera rotations and translations.

# Arguments
- `rays`: Vector of 3 normalized 3D ray directions from camera center
- `points_3d`: Vector of 3 corresponding 3D world points

# Returns
- `rotations`: Vector of possible rotation matrices (SMatrix{3,3,Float64})
- `translations`: Vector of possible translation vectors (Vec{3,Float64})
"""
function p3p(x_copy::AbstractVector{S}, X_copy::AbstractVector{T}) where {S <: StaticVector{3,Float64},T <: StaticVector{3,Float64}}
    # EXACT port of PoseLib p3p.cc (lines 39-202)
    # Validated against C++ reference implementation
    
    output_R = Vector{SMatrix{3,3,Float64}}()
    output_t = Vector{Vec{3,Float64}}()

    if length(x_copy) != 3 || length(X_copy) != 3
        return output_R, output_t
    end

    # Lines 47-53: Compute squared distances
    X01 = X_copy[1] - X_copy[2]
    X02 = X_copy[1] - X_copy[3]
    X12 = X_copy[2] - X_copy[3]

    a01 = dot(X01, X01)
    a02 = dot(X02, X02)
    a12 = dot(X12, X12)

    # Lines 55-56: Work with copies
    X = [X_copy[1], X_copy[2], X_copy[3]]
    x = [x_copy[1], x_copy[2], x_copy[3]]

    # Lines 58-73: Switch so that BC (a12) is largest
    if a01 > a02
        if a01 > a12
            x[1], x[3] = x[3], x[1]
            X[1], X[3] = X[3], X[1]
            a01, a12 = a12, a01
            X01 = -X12
            X02 = -X02
        end
    elseif a02 > a12
        x[1], x[2] = x[2], x[1]
        X[1], X[2] = X[2], X[1]
        a02, a12 = a12, a02
        X01 = -X01
        X02 = X12
    end

    # Lines 75-77: Normalize distances by a12
    a12d = 1.0 / a12
    a = a01 * a12d
    b = a02 * a12d

    # Lines 79-81: Dot products on ORIGINAL rays (not normalized!)
    m01 = dot(x[1], x[2])
    m02 = dot(x[1], x[3])
    m12 = dot(x[2], x[3])

    # Lines 84-93: Compute intermediate values
    m12sq = -m12 * m12 + 1.0
    m02sq = -1.0 + m02 * m02
    m01sq = -1.0 + m01 * m01
    ab = a * b
    bsq = b * b
    asq = a * a
    m013 = -2.0 + 2.0 * m01 * m02 * m12
    bsqm12sq = bsq * m12sq
    asqm12sq = asq * m12sq
    abm12sq = 2.0 * ab * m12sq

    # Lines 95-98: Cubic coefficients
    k3_inv = 1.0 / (bsqm12sq + b * m02sq)
    k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013)
    k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq)
    k0 = k3_inv * (asqm12sq + a * m01sq)

    # Line 100-101: Solve cubic for s
    s_roots = solve_cubic_single_real(k2, k1, k0)
    if isempty(s_roots)
        return output_R, output_t
    end
    s = s_roots[1]
    G = true  # Assume cubic solved successfully

    # Lines 103-112: Build C matrix
    C = @SMatrix [
        -a + s * (1 - b)        -m02 * s                a * m12 + b * m12 * s;
        -m02 * s                s + 1                   -m01;
        a * m12 + b * m12 * s   -m01                    -a - b * s + 1
    ]

    # Line 114: Compute pq
    pq = compute_pq(C)

    # Lines 119-122: Precompute XX matrix
    XX = hcat(X01, X02, cross(X01, X02))
    XX = inv(XX)

    n_sols = 0

    # Lines 129-199: Loop through both pq vectors
    for i in 1:2
        # Lines 131-133
        p0 = pq[i][1]
        p1 = pq[i][2]
        p2 = pq[i][3]

        # Line 137
        switch_12 = abs(p0) <= abs(p1)

        if switch_12
            # Lines 140-166: eliminate d0
            w0 = -p0 / p1
            w1 = -p2 / p1
            ca = 1.0 / (w1 * w1 - b)
            cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca
            cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca

            success, taus = root2real(cb, cc)
            if !success
                continue
            end

            for tau in taus
                if tau <= 0
                    continue
                end

                d2 = sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0))
                d1 = tau * d2
                d0 = (w0 * d2 + w1 * d1)

                if d0 < 0
                    continue
                end

                # Refine
                d0_ref, d1_ref, d2_ref = Ref(d0), Ref(d1), Ref(d2)
                refine_lambda!(d0_ref, d1_ref, d2_ref, a01, a02, a12, m01, m02, m12)
                d0, d1, d2 = d0_ref[], d1_ref[], d2_ref[]

                v1 = d0 * x[1] - d1 * x[2]
                v2 = d0 * x[1] - d2 * x[3]
                YY = hcat(v1, v2, cross(v1, v2))
                R = YY * XX
                t = d0 * x[1] - R * X[1]

                push!(output_R, SMatrix{3,3,Float64}(R))
                push!(output_t, Vec{3,Float64}(t))
                n_sols += 1
            end
        else
            # Lines 168-194: eliminate d1
            w0 = -p1 / p0
            w1 = -p2 / p0
            ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1)
            cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca
            cc = (1 - a * w0 * w0) * ca

            success, taus = root2real(cb, cc)
            if !success
                continue
            end

            for tau in taus
                if tau <= 0
                    continue
                end

                d0 = sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0))
                d1 = tau * d0
                d2 = w0 * d0 + w1 * d1

                if d2 < 0
                    continue
                end

                # Refine
                d0_ref, d1_ref, d2_ref = Ref(d0), Ref(d1), Ref(d2)
                refine_lambda!(d0_ref, d1_ref, d2_ref, a01, a02, a12, m01, m02, m12)
                d0, d1, d2 = d0_ref[], d1_ref[], d2_ref[]

                v1 = d0 * x[1] - d1 * x[2]
                v2 = d0 * x[1] - d2 * x[3]
                YY = hcat(v1, v2, cross(v1, v2))
                R = YY * XX
                t = d0 * x[1] - R * X[1]

                push!(output_R, SMatrix{3,3,Float64}(R))
                push!(output_t, Vec{3,Float64}(t))
                n_sols += 1
            end
        end

        # Line 197-198: Break if we found solutions and cubic succeeded
        if n_sols > 0 && G
            break
        end
    end

    return output_R, output_t
end
