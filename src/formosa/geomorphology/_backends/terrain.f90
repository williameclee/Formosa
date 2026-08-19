!> Computes terrain isolation for digital elevation model rasters.
!!
!! This module implements the native backend used by the public
!! Python terrain API. A maximum-elevation pyramid accelerates exact
!! nearest-higher searches and identifies searches censored by the
!! outer raster footprint.
!!
!! Created: 2026-08-19, En-Chi Lee
module terrain
    use iso_c_binding, only: c_int8_t
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW
    use utils, only: fill_offset_lookup, find_noflow_code, &
                     array2d_oob, mask2ij
    use distances, only: l2dist_xy, l2dist2_xy
    implicit none(type, external)

    private
    public :: compute_isolation

    !> Stores maximum elevations for a level of a spatial pyramid.
    type :: pyramid_level
        real, allocatable :: zmax(:, :)
            !! Maximum valid elevation per block
        integer :: block_size = 1
            !! Block width in original DEM cells
    end type pyramid_level

    !> Describes the maximum-elevation pyramid for a DEM.
    type :: elevation_pyramid
        type(pyramid_level), allocatable :: levels(:)
            !! Pyramid levels
        integer :: dem_nrows
            !! Number of rows in the source DEM
        integer :: dem_ncols
            !! Number of columns in the source DEM
        real :: invalid_value
            !! Sentinel used for invalid leaf cells
        integer :: factor
            !! Reduction factor between adjacent levels
    end type elevation_pyramid
contains
    !> Builds a maximum-elevation pyramid from a DEM and validity
    !! mask.
    !!
    !! Level 0 contains individual DEM cells. Each higher level
    !! stores the maximum elevation in a 'fac' by 'fac' group of
    !! child blocks. Invalid leaf cells receive a negative sentinel
    !! so they cannot qualify as isolation limit points.
    pure subroutine build_elevation_pyramid( &
        z, valids, p, fac, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid.
        logical(kind=1), intent(in) :: valids(:, :)
            !! Validity mask
        integer, intent(in) :: fac
            !! Reduction factor between levels
        type(elevation_pyramid), intent(out) :: p
            !! Constructed pyramid
        integer, intent(out) :: err_code
            !! Backend status code
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

    !> Returns the distance from a cell centre to a raster
    !! footprint.
    !!
    !! The footprint extends half a cell beyond the centres of the
    !! outer cells. irange and jrange give its inclusive cell-index
    !! bounds.
    pure real function min_dist2boundary(ci, cj, irange, jrange, dx, dy) result(dist)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ci, cj
            !! Query-cell row and column
        integer, intent(in) :: irange(2), jrange(2)
            !! Footprint bounds
        real, intent(in) :: dx, dy
            !! Column and row spacing

        dist = min(dy*(min(ci - irange(1), irange(2) - ci) + 0.5), &
                   dx*(min(cj - jrange(1), jrange(2) - cj) + 0.5))
    end function min_dist2boundary

    !> Returns the squared minimum distance from a cell to a block.
    !!
    !! The query-to-block distance is 0 inside the block.
    !! Otherwise, it is the physical distance to the closest cell
    !! centre in the block, accounting for anisotropic grid spacing.
    pure real function min_dist2block( &
        p, lvl, bi, bj, ci, cj, dx, dy) result(dist2)
        implicit none(type, external)
        ! Arguments
        type(elevation_pyramid), intent(in) :: p
            !! Elevation pyramid
        integer, intent(in) :: lvl
            !! Pyramid level
        integer, intent(in) :: bi, bj
            !! Block row and column
        integer, intent(in) :: ci, cj
            !! Query-cell row and column
        real, intent(in) :: dx, dy
            !! Column and row spacing
        ! Local variables
        integer :: bsize
            !! Block width in original DEM cells
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

    !> Finds nearby isolation limit points using fixed cell offsets.
    !!
    !! This preliminary scan supplies an upper distance bound for
    !! the recursive pyramid search. Existing shorter candidates
    !! are kept, so the result is exact for the offsets that are
    !! examined.
    subroutine find_neighbour_ilp( &
        z, valids, isos, offsets, ilp_is, ilp_js, dx, dy)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(:, :)
            !! Validity mask
        integer, intent(in) :: offsets(:, :)
            !! Row-column offsets
        real, intent(in) :: dx, dy
            !! Column and row spacing
        ! Outputs
        real, intent(inout) :: isos(:, :)
            !! Best isolation distances
        integer, intent(inout) :: ilp_is(:, :), ilp_js(:, :)
            !! Best ILP row and column indices
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

    !> Searches a pyramid block recursively for the nearest higher cell.
    !!
    !! A block is pruned when its maximum elevation is not strictly
    !! higher than cz, or when its minimum possible distance exceeds
    !! the best candidate. At level 0, qualifying cells update the
    !! current squared-distance bound and ILP indices.
    pure recursive subroutine search_ilp( &
        p, lvl, bi, bj, ci, cj, cz, dx, dy, best_dist2, best_i, best_j)
        implicit none(type, external)
        ! Arguments
        type(elevation_pyramid), intent(in) :: p
            !! Elevation pyramid
        integer, intent(in) :: lvl
            !! Pyramid level
        integer, intent(in) :: bi, bj
            !! Block row and column
        integer, intent(in) :: ci, cj
            !! Query-cell row and column
        real, intent(in) :: cz
            !! Query-cell elevation
        real, intent(in) :: dx, dy
            !! Column and row spacing
        ! Output
        real, intent(inout) :: best_dist2
            !! Best squared distance
        integer, intent(inout) :: best_i, best_j
            !! Best ILP indices
        ! Local variables
        integer :: sbif, sbil, sbjf, sbjl
            !! Block range at the child level
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

    !> Calculates terrain isolation and footprint censoring for a DEM.
    !!
    !! For every valid cell, isolation is the physical distance to the
    !! nearest valid cell with a strictly higher elevation. The
    !! ilp_is and ilp_js arrays identify that isolation limit point.
    !! Cells without an ILP retain 0 and indices of -1.
    !!
    !! A valid result is censored when its search circle crosses the
    !! outer half-cell raster footprint before reaching the ILP. A
    !! valid cell without an ILP is always censored. Internal
    !! invalid cells do not define observation-window boundaries.
    subroutine compute_isolation( &
        z, valids, isos, ilp_is, ilp_js, censored, nrows, ncols, &
        dx, dy, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! DEM dimensions
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask; false denotes no-data
        real, intent(in) :: dx, dy
            !! Column and row spacing
        ! Outputs
        real, intent(out) :: isos(nrows, ncols)
            !! Isolation distances
        integer, intent(out) :: ilp_is(nrows, ncols), ilp_js(nrows, ncols)
            !! 1-based ILP row and column indices
        logical(kind=1), intent(out) :: censored(nrows, ncols)
            !! Outer-footprint censoring mask
        integer, intent(out) :: err_code
            !! Backend status code
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

        ! Scan immediate neighbours
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

        ! Mark searches truncated by the outer raster footprint
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
    end subroutine compute_isolation
end module terrain
