module terrain
    use iso_c_binding, only: c_int8_t
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW
    use utils, only: fill_offset_lookup, find_noflow_code, &
                     array2d_oob, mask2ij
    use distances, only: l2dist_xy, l2dist2_xy
    implicit none(type, external)

    private
    public :: calculate_isolation

    type :: pyramid_level
        real, allocatable :: zmax(:, :)
        integer :: block_size = 1
    end type pyramid_level

    type :: elevation_pyramid
        type(pyramid_level), allocatable :: levels(:)
        integer :: dem_nrows
        integer :: dem_ncols
        real :: invalid_value
            !! Some really negative value to mark invalid cells
        integer :: factor
            !! How much smaller each layer above should become
    end type elevation_pyramid
contains
    pure subroutine build_elevation_pyramid( &
        z, valids, p, fac, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(:, :)
        integer, intent(in) :: fac
        type(elevation_pyramid), intent(out) :: p
        integer, intent(out) :: err_code
        ! Local variables
        integer :: nrows, ncols
        integer :: cif, cil, cjf, cjl, pi, pj
        integer :: ilvl, nlvls
        integer :: alloc_stat

        err_code = ERR_NO_ERROR
        if (fac < 2) then
            err_code = ERR_INVALID_INPUT
            return
        elseif (any(shape(z) /= shape(valids))) then
            err_code = ERR_INVALID_INPUT
            return
        end if

        ! Count number of levels required
        nrows = size(z, dim=1)
        ncols = size(z, dim=2)
        nlvls = 0
        do while (nrows > 1 .or. ncols > 1)
            nlvls = nlvls + 1
            nrows = (nrows + fac - 1)/fac
            ncols = (ncols + fac - 1)/fac
        end do

        allocate (p%levels(0:nlvls), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Initialise pyramid
        p%factor = fac
        p%invalid_value = -huge(0.)
        p%levels(0)%block_size = 1
        p%dem_nrows = size(z, dim=1)
        p%dem_ncols = size(z, dim=2)
        allocate (p%levels(0)%zmax(p%dem_nrows, p%dem_ncols), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        p%levels(0)%zmax = z
        where (.not. valids)
            p%levels(0)%zmax = p%invalid_value
        end where

        ! Build parent level
        do ilvl = 1, nlvls
            p%levels(ilvl)%block_size = &
                p%levels(ilvl - 1)%block_size*fac
            allocate (p%levels(ilvl)%zmax( &
                      (size(p%levels(ilvl - 1)%zmax, dim=1) + fac - 1)/fac, &
                      (size(p%levels(ilvl - 1)%zmax, dim=2) + fac - 1)/fac), &
                      stat=alloc_stat)
            if (alloc_stat /= 0) then
                err_code = ERR_ALLOCATION_FAILURE
                return
            end if
            ! Fill the zmax
            do pj = 1, size(p%levels(ilvl)%zmax, dim=2)
            do pi = 1, size(p%levels(ilvl)%zmax, dim=1)
                ! Find the extent of the parent
                cif = fac*(pi - 1) + 1
                cil = min(fac*pi, size(p%levels(ilvl - 1)%zmax, dim=1))
                cjf = fac*(pj - 1) + 1
                cjl = min(fac*pj, size(p%levels(ilvl - 1)%zmax, dim=2))
                p%levels(ilvl)%zmax(pi, pj) = &
                    maxval(p%levels(ilvl - 1)%zmax(cif:cil, cjf:cjl))
            end do
            end do
        end do
    end subroutine build_elevation_pyramid

    pure real function min_dist2boundary(ci, cj, irange, jrange, dx, dy) result(dist)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ci, cj
        integer, intent(in) :: irange(2), jrange(2)
        real, intent(in) :: dx, dy

        dist = min(dy*(min(ci - irange(1), irange(2) - ci) + 0.5), &
                   dx*(min(cj - jrange(1), jrange(2) - cj) + 0.5))
    end function min_dist2boundary

    pure real function min_dist2block( &
        p, lvl, bi, bj, ci, cj, dx, dy) result(dist2)
        implicit none(type, external)
        ! Arguments
        type(elevation_pyramid), intent(in) :: p
        integer, intent(in) :: lvl
            !! Level in the pyramid
        integer, intent(in) :: bi, bj
            !! Index of the block within the pyramid level
        integer, intent(in) :: ci, cj
            !! Index of the cell to calculate distance against
        real, intent(in) :: dx, dy
        ! Local variables
        integer :: bsize
            !! Size of the block
        integer :: di, dj
            !! Index offset between the cell to check and the
            !! closest cell in the block
        integer :: ifirst, ilast, jfirst, jlast
            !! Index range of the block at the original DEM level

        bsize = p%levels(lvl)%block_size

        ! Find the range of the box
        ifirst = (bi - 1)*bsize + 1
        ilast = min(bi*bsize, p%dem_nrows)
        jfirst = (bj - 1)*bsize + 1
        jlast = min(bj*bsize, p%dem_ncols)
        ! Calculate offset and distance
        di = max(0, ifirst - ci, ci - ilast)
        dj = max(0, jfirst - cj, cj - jlast)
        dist2 = dy**2*di**2 + dx**2*dj**2
    end function min_dist2block

    subroutine find_neighbour_ilp( &
        z, valids, isos, offsets, ilp_is, ilp_js, dx, dy)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(:, :)
            !! Validity mask (false for no-data)
        integer, intent(in) :: offsets(:, :)
        real, intent(in) :: dx, dy
        ! Outputs
        real, intent(inout) :: isos(:, :)
        integer, intent(inout) :: ilp_is(:, :), ilp_js(:, :)
        ! Local variables
        integer :: ci, cj, ni, nj
        integer :: iofs
        real :: dist

        !$omp PARALLEL DO COLLAPSE(2)&
        !$omp DEFAULT(SHARED) PRIVATE(cj, ci, ni, nj, iofs, dist) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, size(z, dim=2)
        do ci = 1, size(z, dim=1)
            if (.not. valids(ci, cj)) cycle
            if (isos(ci, cj) > 0) cycle
            do iofs = 1, size(offsets, dim=2)
                ni = ci + offsets(1, iofs)
                nj = cj + offsets(2, iofs)
                ! Check bounds
                if (array2d_oob(ni, nj, size(z, dim=1), size(z, dim=2))) cycle
                ! Check if neighbour is part of the same flat
                if (.not. valids(ni, nj)) cycle
                ! Record neighbour with higher elevation
                if (z(ni, nj) <= z(ci, cj)) cycle
                dist = l2dist_xy(dy*ci, dx*cj, dy*ni, dx*nj)
                if ((isos(ci, cj) > 0) .and. &
                    (dist >= isos(ci, cj))) cycle
                isos(ci, cj) = dist
                ilp_is(ci, cj) = ni
                ilp_js(ci, cj) = nj
            end do
        end do
        end do
        !$omp END PARALLEL DO
    end subroutine

    pure recursive subroutine search_ilp( &
        p, lvl, bi, bj, ci, cj, cz, dx, dy, best_dist2, best_i, best_j)
        implicit none(type, external)
        ! Arguments
        type(elevation_pyramid), intent(in) :: p
        integer, intent(in) :: lvl
            !! Level in the pyramid
        integer, intent(in) :: bi, bj
            !! Index of the block within the pyramid level
        integer, intent(in) :: ci, cj
            !! Index of the cell to calculate distance against
        real, intent(in) :: cz
            !! Elevation of the cell
        real, intent(in) :: dx, dy
        ! Output
        real, intent(inout) :: best_dist2
        integer, intent(inout) :: best_i, best_j
        ! Local variables
        integer :: sbif, sbil, sbjf, sbjl
            !! Range of the block in at its children's level
        integer :: sbi, sbj
        real :: dist2

        ! Return if no potential ILPs
        if (p%levels(lvl)%zmax(bi, bj) <= cz) return
        ! Return if too far
        if ((best_dist2 >= 0) .and. &
            (min_dist2block(p, lvl, bi, bj, ci, cj, dx, dy) > best_dist2)) return

        if (lvl == 0) then
            dist2 = l2dist2_xy(dy*ci, dx*cj, dy*bi, dx*bj)
            if ((best_dist2 >= 0) .and. (dist2 >= best_dist2)) return
            ! A potential ILP found
            best_dist2 = dist2
            best_i = bi
            best_j = bj
            return
        end if

        ! Search the children of the block
        sbif = p%factor*(bi - 1) + 1
        sbil = min(p%factor*bi, size(p%levels(lvl - 1)%zmax, 1))
        sbjf = p%factor*(bj - 1) + 1
        sbjl = min(p%factor*bj, size(p%levels(lvl - 1)%zmax, 2))

        do sbi = sbif, sbil
        do sbj = sbjf, sbjl
            call search_ilp(p, lvl - 1, sbi, sbj, ci, cj, cz, dx, dy, best_dist2, best_i, best_j)
        end do
        end do
    end subroutine search_ilp

    subroutine calculate_isolation( &
        z, valids, isos, ilp_is, ilp_js, censored, nrows, ncols, &
        dx, dy, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        real, intent(in) :: dx, dy
            !! Grid spacing
        ! Outputs
        real, intent(out) :: isos(nrows, ncols)
        integer, intent(out) :: ilp_is(nrows, ncols), ilp_js(nrows, ncols)
            !! Cell indices for the isolation limit point (ILP)
        logical(kind=1), intent(out) :: censored(nrows, ncols)
        integer, intent(out) :: err_code
        ! Local variables
        integer, parameter :: offsets1(2, 8) = reshape( &
                              [0, 1, 1, 0, 0, -1, -1, 0, &
                               1, 1, 1, -1, -1, -1, -1, 1], [2, 8])
        integer, parameter :: offsets2(2, 16) = reshape( &
                              [0, 2, 2, 0, 0, -2, -2, 0, &
                               1, 2, 2, 1, 2, -1, 1, -2, -1, -2, -2, -1, -2, 1, -1, 2, &
                               2, 2, 2, -2, -2, -2, -2, 2], [2, 16])
        integer :: ci, cj
        real :: dist2
        type(elevation_pyramid) :: pyramid
        integer, parameter :: pyramid_factor = 2
        real :: dist_bdry

        err_code = ERR_NO_ERROR
        ilp_is = -1
        ilp_js = -1
        isos = 0

        ! Scan immideate neighbours
        call find_neighbour_ilp( &
            z, valids, isos, offsets1, ilp_is, ilp_js, dx, dy)
        ! Scan secondary neighbours
        call find_neighbour_ilp( &
            z, valids, isos, offsets2, ilp_is, ilp_js, dx, dy)

        ! Find remaining isolation using the pyramid
        call build_elevation_pyramid( &
            z, valids, pyramid, pyramid_factor, err_code)
        if (err_code /= ERR_NO_ERROR) return

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(cj, ci, dist2) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
        do ci = 1, nrows
            if (isos(ci, cj) > 0.0) then
                dist2 = isos(ci, cj)**2
            else
                dist2 = -1.0
            end if

            if (.not. valids(ci, cj)) cycle
            call search_ilp( &
                pyramid, ubound(pyramid%levels, dim=1), &
                1, 1, ci, cj, z(ci, cj), dx, dy, dist2, &
                ilp_is(ci, cj), ilp_js(ci, cj))
            if (dist2 > 0) isos(ci, cj) = sqrt(real(dist2))
        end do
        end do
        !$omp END PARALLEL DO

        ! If the isolation in larger than the cell's distance to
        ! the nearest boundary, it is just an upper bound because
        ! the search area is truncated
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(cj, ci, dist_bdry) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
        do ci = 1, nrows
            dist_bdry = min_dist2boundary(ci, cj, [1, nrows], [1, ncols], dx, dy)
            censored(ci, cj) = &
                valids(ci, cj) .and. &
                (isos(ci, cj) == 0 .or. (dist_bdry < isos(ci, cj)))
        end do
        end do
        !$omp END PARALLEL DO
    end subroutine calculate_isolation
end module terrain
