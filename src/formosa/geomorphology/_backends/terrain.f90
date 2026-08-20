!> Computes terrain isolation for digital elevation model rasters.
!!
!! This module implements the native backend used by the public
!! Python terrain API. A maximum-elevation pyramid accelerates exact
!! nearest-higher searches and identifies searches censored by the
!! outer raster footprint.
!!
!! Created: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
!! Last modified: 2026-08-21, En-Chi Lee (williameclee@gmail.com)
module terrain
    use iso_c_binding, only: c_int8_t, c_int32_t
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW
    use utils, only: array2d_oob, id2ij_checked, ij2id_checked
    use distances, only: l2dist_xy, l2dist2_xy
    implicit none(type, external)

    private

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
    public :: compute_isolation, compute_prominence
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
    pure real function min_dist2boundary( &
        ci, cj, irange, jrange, dx, dy) result(dist)
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
    subroutine find_neighbour_ilp2( &
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
            !! Best isolation distances squared
        integer, intent(inout) :: ilp_is(:, :), ilp_js(:, :)
            !! Best ILP row and column indices
        ! Local variables
        integer :: ci, cj, ni, nj
        integer :: iofs
        real :: dist2

        !$omp PARALLEL DO COLLAPSE(2)&
        !$omp DEFAULT(SHARED) PRIVATE(cj, ci, ni, nj, iofs, dist2) &
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
                dist2 = l2dist2_xy(dy*ci, dx*cj, dy*ni, dx*nj)
                if ((isos(ci, cj) > 0) .and. &
                    (dist2 >= isos(ci, cj))) cycle
                isos(ci, cj) = dist2
                ilp_is(ci, cj) = ni
                ilp_js(ci, cj) = nj
                exit
            end do
        end do
        end do
        !$omp END PARALLEL DO
    end subroutine

    !> Searches a pyramid block recursively for the nearest higher
    !! cell.
    !!
    !! A block is pruned when its maximum elevation is not strictly
    !! higher than cz, or when its minimum possible distance exceeds
    !! the best candidate. At level 0, qualifying cells update the
    !! current squared-distance bound and ILP indices.
    pure recursive subroutine search_ilp2( &
        p, lvl, bi, bj, ci, cj, cz, dx, dy, &
        best_dist2, best_i, best_j)
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
            call search_ilp2(p, lvl - 1, sbi, sbj, ci, cj, cz, &
                             dx, dy, best_dist2, best_i, best_j)
        end do
        end do
    end subroutine search_ilp2

    !> Calculates terrain isolation and footprint censoring for a
    !! DEM.
    !!
    !! For every valid cell, isolation is the physical distance to
    !! the nearest valid cell with a strictly higher elevation. The
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
        call find_neighbour_ilp2( &
            z, valids, isos, offsets1, ilp_is, ilp_js, dx, dy)
        ! Scan secondary neighbours
        call find_neighbour_ilp2( &
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
                dist2 = isos(ci, cj)
            else
                dist2 = -1.0
            end if

            if (.not. valids(ci, cj)) cycle
            call search_ilp2( &
                pyramid, ubound(pyramid%levels, dim=1), &
                1, 1, ci, cj, z(ci, cj), dx, dy, dist2, &
                ilp_is(ci, cj), ilp_js(ci, cj))
            if (dist2 > 0) isos(ci, cj) = dist2
        end do
        end do
        !$omp END PARALLEL DO

        ! Convert from distance squared to distance
        isos = sqrt(isos)

        ! Mark searches truncated by the outer raster footprint
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(cj, ci, dist_bdry) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
        do ci = 1, nrows
            dist_bdry = min_dist2boundary( &
                        ci, cj, [1, nrows], [1, ncols], dx, dy)
            censored(ci, cj) = &
                valids(ci, cj) .and. &
                (isos(ci, cj) == 0 .or. (dist_bdry < isos(ci, cj)))
        end do
        end do
        !$omp END PARALLEL DO
    end subroutine compute_isolation

    subroutine find_connections( &
        cids, labels, nlabels, offsets, nrows, ncols, &
        samez_start, samez_end, order_pos, err_code)
        implicit none(type, external)
        ! Arguments
        integer, contiguous, intent(in) :: cids(:)
        integer, intent(in) :: offsets(:, :)
            !! List of offsets for each flow direction
        integer, intent(in) :: nrows, ncols
        integer, intent(in) :: order_pos(:)
        integer, intent(in) :: samez_start, samez_end
        !! DEM dimensions
        ! Outputs
        integer, contiguous, intent(out) :: labels(:)
        integer, intent(out) :: nlabels
            !! Number of connected components
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: icell, ncell
        integer :: ci, cj, cid, ni, nj, nid
        integer :: iofs
        logical(kind=1) :: is_valid
        integer :: queue(size(cids))
        integer :: nqueued, iqueue, nprocessed

        ! Initialise
        err_code = ERR_NO_ERROR
        labels = 0
        nlabels = 0
        nprocessed = 0
        do icell = 1, size(cids)
            ! Find the next unprocessed cell
            if (labels(icell) /= 0) cycle
            nlabels = nlabels + 1
            labels(icell) = nlabels
            ! Push the seed to the queue
            nqueued = 1
            queue(nqueued) = icell
            ! Flood fill from the seed
            iqueue = 1
            do while (iqueue <= nqueued)
                cid = cids(queue(iqueue))
                ! Check if its neighbours are in the queue
                call id2ij_checked(cid, nrows, ncols, ci, cj, is_valid)
                if (.not. is_valid) then
                    err_code = ERR_INVALID_INPUT
                    return
                end if
                ! Check all the neighbours
                do iofs = 1, size(offsets, dim=1)
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    nid = ij2id_checked(ni, nj, nrows, ncols)
                    if (nid == 0) cycle
                    ncell = order_pos(nid)
                    if (ncell < samez_start .or. ncell > samez_end) cycle
                    ncell = ncell - samez_start + 1

                    if (labels(ncell) /= 0) cycle
                    labels(ncell) = nlabels
                    nqueued = nqueued + 1
                    queue(nqueued) = ncell
                end do
                iqueue = iqueue + 1
            end do
        end do
    end subroutine find_connections

    recursive function find_root_peak(parent_peaks, peak) &
        result(root_peak)
        integer, intent(inout) :: parent_peaks(:)
        integer, intent(in) :: peak
        integer :: root_peak

        if (parent_peaks(peak) /= peak) then
            parent_peaks(peak) = &
                find_root_peak(parent_peaks, parent_peaks(peak))
        end if
        root_peak = parent_peaks(peak)
    end function find_root_peak

    subroutine find_connection_higher_grounds( &
        peaks, cids, labels, label, offsets, &
        parent_peaks, higher_peaks, n_higher_peaks, nrows, ncols, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: peaks(nrows, ncols)
        integer, contiguous, intent(in) :: cids(:)
        integer, contiguous, intent(in) :: labels(:)
        integer, intent(in) :: label
        integer, intent(in) :: offsets(:, :)
            !! List of offsets for each flow direction
        integer, intent(in) :: nrows, ncols
            !! DEM dimensions
        ! Outputs
        integer, intent(inout) :: parent_peaks(:)
        integer, intent(out) :: higher_peaks(:)
        integer, intent(out) :: n_higher_peaks
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: ci, cj, cid, ni, nj
        integer :: icell
        integer :: iofs
        integer :: peak
        logical(kind=1) :: is_valid

        err_code = ERR_NO_ERROR
        n_higher_peaks = 0

        do icell = 1, size(cids)
            if (labels(icell) /= label) cycle
            cid = cids(icell)
            call id2ij_checked(cid, nrows, ncols, ci, cj, is_valid)
            if (.not. is_valid) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            do iofs = 1, size(offsets, dim=1)
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                peak = peaks(ni, nj)
                if (peak == 0) cycle
                peak = find_root_peak(parent_peaks, peak)
                ! Skip already recorded areas
                if (any(higher_peaks(1:n_higher_peaks) == peak)) cycle
                n_higher_peaks = n_higher_peaks + 1
                higher_peaks(n_higher_peaks) = peak
            end do
        end do
    end subroutine find_connection_higher_grounds

    subroutine process_single_label_area( &
        z, cells, proms, offsets, &
        peaks, labels, ilabel, higher_peaks, copeaks, &
        parent_peaks, samez, samez_start, samez_end, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(*)
        integer, intent(in) :: cells(:)
        integer, intent(in) :: offsets(:, :)
        integer, intent(in) :: labels(:)
        integer, intent(in) :: ilabel
        real, intent(in) :: samez
        integer, intent(in) :: samez_start, samez_end
        ! Outputs
        real, intent(inout) :: proms(*)
        integer, intent(inout) :: peaks(:, :)
        integer, intent(inout) :: higher_peaks(:)
        integer, intent(inout) :: copeaks(:)
        integer, intent(inout) :: parent_peaks(:)
        integer, intent(out) :: err_code
        ! Local variables
        integer :: nrows, ncols
        integer :: ipeak, n_higher_peaks
        integer :: peak, copeak, peak2keep
        integer :: icell
        integer :: ci, cj
        integer :: label
        real :: zpeak2keep, zpeak
        logical(kind=1) :: is_valid

        err_code = ERR_NO_ERROR

        nrows = size(peaks, dim=1)
        ncols = size(peaks, dim=2)

        ! Find all higher cells (i.e. processed) connected to the
        ! region, and which peak each correspond to
        call find_connection_higher_grounds( &
            peaks, &
            cells(samez_start:samez_end), &
            labels(samez_start:samez_end), &
            ilabel, offsets, &
            parent_peaks, higher_peaks, n_higher_peaks, &
            nrows, ncols, err_code)
        if (err_code /= ERR_NO_ERROR) return

        ! Process new peaks or connecting pieces to existing peaks
        if (n_higher_peaks == 0) then
            ! If no higher areas, this is a new peak
            peak = 0
            ! Record the new peak with the largest icell
            do icell = samez_end, samez_start, -1
                if (labels(icell) /= ilabel) cycle
                if (peak == 0) peak = icell
                call id2ij_checked( &
                    cells(icell), nrows, ncols, ci, cj, is_valid)
                peaks(ci, cj) = peak
                ! Set the prominence as -1 so that we can identify surviving peaks with unknown prominence
                proms(cells(icell)) = -1
            end do
            return
        elseif (n_higher_peaks == 1) then
            ! If only 1 higher area, merge with it
            do icell = samez_start, samez_end
                if (labels(icell) /= ilabel) cycle
                call id2ij_checked( &
                    cells(icell), nrows, ncols, ci, cj, is_valid)
                peaks(ci, cj) = higher_peaks(1)
            end do
            return
        end if

        ! Merge peaks
        ! Find the peak to retain
        peak2keep = higher_peaks(1)
        zpeak2keep = z(cells(peak2keep))
        do ipeak = 2, n_higher_peaks
            peak = higher_peaks(ipeak)
            if (z(cells(peak)) < zpeak2keep) cycle
            peak2keep = peak
            zpeak2keep = z(cells(peak))
        end do

        ! Update all other peaks
        do ipeak = 1, n_higher_peaks
            peak = find_root_peak(parent_peaks, higher_peaks(ipeak))
            if (peak == peak2keep) cycle
            ! Process co-winning peaks
            if (zpeak2keep == z(cells(peak))) then
                copeak = peak2keep
                do while (copeaks(copeak) /= 0)
                    copeak = copeaks(copeak)
                end do
                parent_peaks(peak) = peak2keep
                copeaks(copeak) = peak
                cycle
            end if
            ! Process lost peaks to be merged
            ! (including their co-peaks)
            do while (peak /= 0)
                parent_peaks(peak) = peak2keep
                ! Mark the whole peak region
                icell = peak
                label = labels(icell)
                zpeak = z(cells(peak))
                do while (icell >= 1)
                    if (z(cells(icell)) /= zpeak) exit
                    if (labels(icell) == label) then
                        proms(cells(icell)) = zpeak - samez
                    end if
                    icell = icell - 1
                end do
                ! Go to the next copeak
                peak = copeaks(peak)
            end do
        end do
        ! Merge the current saddle too
        do icell = samez_start, samez_end
            if (labels(icell) /= ilabel) cycle
            call id2ij_checked( &
                cells(icell), nrows, ncols, ci, cj, is_valid)
            peaks(ci, cj) = peak2keep
        end do
    end subroutine process_single_label_area

    subroutine compute_prominence( &
        z, orders, proms, nrows, ncols, nvalids, &
        offsets, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! DEM dimensions
        integer, intent(in) :: nvalids
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        integer(c_int32_t), intent(in) :: orders(nvalids)
            !! Cell linear index sorted in increasing elevation
            !! order (they will be processed in reversed order)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        real, intent(out) :: proms(nrows, ncols)
            !! Topographic prominence height grid
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: icell
        integer :: ci, cj
        logical(kind=1) :: is_valid
        real :: samez
        integer :: samez_start, samez_end
            !! Where in the 'rev_orders' queue that cells of the
            !! same elevation 'samez' is stored
        integer(c_int32_t), allocatable :: labels(:)
        integer :: ilabel, nlabels
        integer(c_int32_t), allocatable :: peaks(:, :)
        integer(c_int32_t), allocatable :: parent_peaks(:)
        integer(c_int32_t), allocatable :: higher_peaks(:)
        integer(c_int32_t), allocatable :: copeaks(:)
        integer(c_int32_t), allocatable :: order_pos(:)
        integer :: alloc_stat

        err_code = ERR_NO_ERROR
        allocate ( &
            labels(nvalids), &
            peaks(nrows, ncols), parent_peaks(nvalids), &
            higher_peaks(nvalids), &
            copeaks(nvalids), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        proms = 0
        peaks = 0
        copeaks = 0

        allocate (order_pos(nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        order_pos = 0
        do icell = 1, nvalids
            if (orders(icell) < 1 .or. orders(icell) > nrows*ncols) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            parent_peaks(icell) = icell
            ! Build inverse lookup from linear ID in 'dem' to position in 'orders'
            order_pos(orders(icell)) = icell
        end do

        icell = nvalids
        do while (icell >= 1)
            call id2ij_checked( &
                orders(icell), nrows, ncols, ci, cj, is_valid)
            if (.not. is_valid) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            samez = z(ci, cj)
            samez_end = icell
            samez_start = icell
            do while (icell >= 1)
                call id2ij_checked( &
                    orders(icell), nrows, ncols, ci, cj, is_valid)
                if (.not. is_valid) then
                    err_code = ERR_INVALID_INPUT
                    return
                end if
                if (z(ci, cj) /= samez) exit
                icell = icell - 1
            end do
            samez_start = icell + 1

            ! group connected regions of the same elevation
            call find_connections( &
                orders(samez_start:samez_end), &
                labels(samez_start:samez_end), &
                nlabels, offsets, nrows, ncols, &
                samez_start, samez_end, order_pos, err_code)
            if (err_code /= ERR_NO_ERROR) return
            ! Process each connected regions one at a time
            do ilabel = 1, nlabels
                call process_single_label_area( &
                    z, orders, proms, offsets, &
                    peaks, labels, ilabel, higher_peaks, copeaks, &
                    parent_peaks, samez, samez_start, samez_end, &
                    err_code)
                if (err_code /= ERR_NO_ERROR) return
            end do
        end do
    end subroutine compute_prominence
end module terrain
