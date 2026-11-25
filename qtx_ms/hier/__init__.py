"""Minimal density-matrix hierarchy for the open-boundary Schroedinger equation (user, 2026-11-25; a fresh, small version of
qtx_ms.mlevel): real symmetric H, S; GPU or CPU; one rank or MPI / NCCL.

    plan    core graph -> dependence tree: levels, windows, which block pair is computed where (once) and who needs it
    obc     lead Sigma(z) and the eigenpairs of Gamma, on the GPU, parallel over the real energies and the contour points
    fine    sparse device rows on the device: dense blocks, fine operator, fine windows
    levels  bases, S_k and H_k of the block pairs (real, energy independent, formed once for all energies), smoothers, cycle
    run     driver:  python -m qtx_ms.hier.run --inputs DIR --dE 0.15,0.30 ...

Only Sigma, the window factors and the bottom LU depend on the energy. H, S real and Sigma complex symmetric: rho of a window is
real symmetric, so every basis U_k and every S_k = U_k^T S_{k-1} U_k is real."""
