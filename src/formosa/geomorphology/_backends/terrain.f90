!> Computes terrain isolation and topographic prominence for DEM
!! rasters.
!!
!! This module implements the native backend used by the public
!! Python terrain API. It provides accelerated isolation searches
!! via maximum-elevation pyramids and topological sweep-plane
!! prominence computations.
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

    !> Labels connected components (plateaus) of equal elevation.
    !!
    !! Performs a breadth-first search (BFS) flood fill to identify
    !! connected flat components among cells sharing the same
    !! elevation within the current slice range
    !! [slice_start, slice_end].
    pure subroutine label_plateaus( &
        cids, labels, nlabels, offsets, nrows, ncols, &
        slice_start, slice_end, order_lookup, err_code)
        implicit none(type, external)
        ! Arguments
        integer, contiguous, intent(in) :: cids(:)
            !! Slice of cell linear IDs for the current elevation
        integer, intent(in) :: offsets(:, :)
            !! Row and column offsets for neighbour connectivity
        integer, intent(in) :: nrows, ncols
            !! DEM raster dimensions
        integer, intent(in) :: order_lookup(:)
            !! Inverse lookup mapping linear cell ID to position in
            !! all of cids (bedyond this slice)
        integer, intent(in) :: slice_start, slice_end
            !! Index bounds of the current elevation slice in all of
            !! cids
        ! Outputs
        integer, contiguous, intent(out) :: labels(:)
            !! Output connected component label per cell in the
            !! slice (1 to nlabels)
        integer, intent(out) :: nlabels
            !! Number of connected components found
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: icell
            !! Index in 'cids' slice
        integer :: ncell
            !! Index in 'cids' of neighbouring cell
        integer :: ci, cj, ni, nj
            !! Row and column indices of current and neighbour cell
        integer :: cid, nid
            !! Linear ID of current and neighbour cell
        integer :: iofs
            !! neighbour direction offset index
        logical(kind=1) :: is_valid
            !! True if index conversion succeeded
        integer :: queue(size(cids))
            !! FIFO queue for BFS flood fill
        integer :: nqueued
            !! Total number of cells queued
        integer :: iqueue
            !! Current position in BFS queue being processed

        ! Initialise
        err_code = ERR_NO_ERROR
        labels = 0
        nlabels = 0
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
                    ncell = order_lookup(nid)
                    if (ncell < slice_start .or. ncell > slice_end) cycle
                    ncell = ncell - slice_start + 1

                    if (labels(ncell) /= 0) cycle
                    labels(ncell) = nlabels
                    nqueued = nqueued + 1
                    queue(nqueued) = ncell
                end do
                iqueue = iqueue + 1
            end do
        end do
    end subroutine label_plateaus

    !> Finds the canonical root peak for a peak domain in a
    !! disjoint-set forest.
    !!
    !! Implements path compression so that future lookups are
    !! faster.
    recursive function find_root_domain(prnt_doms, dom) &
        result(root_dom)
        ! Arguments
        integer, intent(inout) :: prnt_doms(:)
            !! Disjoint-set parent array tracking peak domain merges
        integer, intent(in) :: dom
            !! Query peak ID (index in 'sorted_cids')
        ! Result
        integer :: root_dom
            !! Canonical root peak ID representing this peak domain

        if (prnt_doms(dom) /= dom) then
            prnt_doms(dom) = &
                find_root_domain(prnt_doms, prnt_doms(dom))
        end if
        root_dom = prnt_doms(dom)
    end function find_root_domain

    !> Identifies unique higher peak domains adjacent to a plateau
    !! component.
    !!
    !! Traverses all cells belonging to a connected flat component,
    !! inspects their spatial neighbours, and records each distinct
    !! adjacent higher peak domain (resolved to its root
    !! representative in the disjoint-set forest).
    subroutine find_adjacent_higher_domains( &
        doms, sorted_cids, label_head, labels, offsets, &
        prnt_doms, higher_doms, n_higher_doms, nrows, ncols, &
        err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: doms(nrows, ncols)
            !! 2D peak domain grid recording owning (non-root) peak
            !! for each cell
        integer, contiguous, intent(in) :: sorted_cids(:)
            !! Linear cell IDs sorted in ascending elevation order
        integer, intent(in) :: label_head
            !! Index of the first cell in this plateau component's
            !! linked list
        integer, contiguous, intent(in) :: labels(:)
            !! Linked-list 'next' pointers chaining cells in the
            !! same component
        integer, intent(in) :: offsets(:, :)
            !! Row and column offsets for neighbour connectivity
        integer, intent(in) :: nrows, ncols
            !! DEM raster dimensions
        ! Outputs
        integer, intent(inout) :: prnt_doms(:)
            !! Disjoint-set parent array tracking peak domain merges
        integer, intent(out) :: higher_doms(:)
            !! Output buffer of unique adjacent higher root peak IDs
        integer, intent(out) :: n_higher_doms
            !! Number of unique adjacent higher root peaks found
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: ci, cj, ni, nj
            !! Row and column indices of current and neighbour cell
        integer :: cid
            !! Linear ID of current cell
        integer :: icell
            !! Current cell index traversing the component linked
            !! list
        integer :: iofs
            !! Neighbour direction offset index
        integer :: dom
            !! Domain ID of neighbour cell resolved to its root
            !! representative
        logical(kind=1) :: is_valid
            !! True if index conversion succeeded

        err_code = ERR_NO_ERROR
        n_higher_doms = 0

        icell = label_head
        do while (icell /= 0)
            cid = sorted_cids(icell)
            call id2ij_checked(cid, nrows, ncols, ci, cj, is_valid)
            if (.not. is_valid) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            do iofs = 1, size(offsets, dim=1)
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                dom = doms(ni, nj)
                if (dom == 0) cycle
                dom = find_root_domain(prnt_doms, dom)
                ! Skip already recorded areas
                if (any(higher_doms(1:n_higher_doms) == dom)) cycle
                n_higher_doms = n_higher_doms + 1
                higher_doms(n_higher_doms) = dom
            end do
            icell = labels(icell)
        end do
    end subroutine find_adjacent_higher_domains

    !> Processes a single connected plateau component during
    !! prominence sweep.
    !!
    !! Handles three morphological cases for the component:
    !! 1. Isolated local maximum (0 higher neighbours): creates a
    !!    new peak domain.
    !! 2. Slope/ridge extension (1 higher neighbour): merges cells
    !!    into that domain.
    !! 3. Saddle/col (>= 2 higher neighbours): identifies the
    !!    winning domain, finalises prominence for all subordinate
    !!    domains (and tied co-peaks), and unions their domains into
    !!    the winning domain.
    subroutine process_plateau( &
        z, sorted_cids, proms, &
        peaks, saddles, npeaks, nsaddles, offsets, &
        doms, label_heads, labels, label, higher_doms, copeaks, &
        prnt_doms, slice_z, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(*)
            !! Flattened DEM elevation array
        integer, contiguous, intent(in) :: sorted_cids(:)
            !! 1-based linear cell IDs sorted in ascending elevation
            !! order
        integer, intent(in) :: offsets(:, :)
            !! Row and column offsets for neighbour connectivity
        integer, contiguous, intent(in) :: labels(:), label_heads(:)
            !! Linked-list 'next' pointers and bucket head indices
            !! per component
        integer, intent(in) :: label
            !! Component index being processed
        real, intent(in) :: slice_z
            !! Elevation of the current horizontal slice/plateau
        ! Outputs
        real, intent(inout) :: proms(*)
            !! Topographic prominence array
        integer, intent(inout) :: peaks(*)
            !! 1-based peak label array for summit cells
        integer, intent(inout) :: saddles(*)
            !! 1-based saddle label array for key saddle cells
        integer, intent(inout) :: npeaks, nsaddles
            !! Running counts of identified peaks and saddles
        integer, intent(inout) :: doms(:, :)
            !! 2D peak domain grid recording owning peak for each
            !! cell
        integer, intent(inout) :: higher_doms(:)
            !! Buffer for adjacent higher peak IDs
        integer, intent(inout) :: copeaks(:)
            !! Singly-linked list tracking tied summits of identical
            !! elevation
        integer, intent(inout) :: prnt_doms(:)
            !! Disjoint-set parent array tracking peak domain merges
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: idom
            !! Index for iterating over adjacent higher peaks
        integer :: n_higher_doms
            !! Number of adjacent higher peaks touching this
            !! component
        integer :: dom
            !! Current peak ID being examined or merged
        integer :: codom
            !! Traversal pointer for the co-peak linked list
        integer :: winner_dom
            !! Dominant peak ID retaining its summit domain at this
            !! saddle
        real :: winner_z
            !! Elevation of the dominant peak
        real :: zpeak
            !! Elevation of a subordinate peak being closed at this
            !! saddle
        integer :: icell
            !! Cell index traversing the component linked list
        integer :: nrows, ncols
            !! DEM raster dimensions
        integer :: ci, cj
            !! Row and column indices of current cell
        logical(kind=1) :: is_valid
            !! True if index conversion succeeded

        err_code = ERR_NO_ERROR

        nrows = size(doms, dim=1)
        ncols = size(doms, dim=2)

        ! Find all higher cells (i.e. processed) connected to the
        ! region, and which peak each correspond to
        call find_adjacent_higher_domains( &
            doms, sorted_cids, label_heads(label), labels, &
            offsets, prnt_doms, higher_doms, n_higher_doms, &
            nrows, ncols, err_code)
        if (err_code /= ERR_NO_ERROR) return

        ! Process new peaks or connecting pieces to existing peaks
        if (n_higher_doms == 0) then
            npeaks = npeaks + 1
            ! Record the new peak with the largest icell
            dom = label_heads(label)
            icell = label_heads(label)
            do while (icell /= 0)
                call id2ij_checked( &
                    sorted_cids(icell), nrows, ncols, ci, cj, is_valid)
                doms(ci, cj) = dom
                ! Set the prominence as -1 so that we can identify
                ! surviving peaks with unknown prominence
                proms(sorted_cids(icell)) = -1
                peaks(sorted_cids(icell)) = npeaks
                ! Go to the next cell with the same label
                icell = labels(icell)
            end do
            return
        elseif (n_higher_doms == 1) then
            ! If only 1 higher area, merge with it
            icell = label_heads(label)
            do while (icell /= 0)
                call id2ij_checked( &
                    sorted_cids(icell), nrows, ncols, ci, cj, is_valid)
                doms(ci, cj) = higher_doms(1)
                ! Go to the next cell with the same label
                icell = labels(icell)
            end do
            return
        end if

        ! Merge domains
        ! Find the domain to retain
        winner_dom = higher_doms(1)
        winner_z = z(sorted_cids(winner_dom))
        do idom = 2, n_higher_doms
            dom = higher_doms(idom)
            if (z(sorted_cids(dom)) < winner_z) cycle
            winner_dom = dom
            winner_z = z(sorted_cids(dom))
        end do

        ! Update all other peaks
        do idom = 1, n_higher_doms
            dom = find_root_domain(prnt_doms, higher_doms(idom))
            if (dom == winner_dom) cycle
            ! Process co-winning peaks
            if (winner_z == z(sorted_cids(dom))) then
                codom = winner_dom
                do while (copeaks(codom) /= 0)
                    codom = copeaks(codom)
                end do
                prnt_doms(dom) = winner_dom
                copeaks(codom) = dom
                cycle
            end if
            ! Process lost peaks to be merged
            ! (including their co-peaks)
            do while (dom /= 0)
                prnt_doms(dom) = winner_dom
                ! Mark the whole peak region
                zpeak = z(sorted_cids(dom))
                icell = dom
                do while (icell /= 0)
                    if (z(sorted_cids(icell)) /= zpeak) exit
                    proms(sorted_cids(icell)) = zpeak - slice_z
                    ! Go to the next cell with the same label
                    icell = labels(icell)
                end do
                ! Go to the next co-peak
                dom = copeaks(dom)
            end do
        end do
        ! Merge the current saddle too
        nsaddles = nsaddles + 1
        icell = label_heads(label)
        do while (icell /= 0)
            call id2ij_checked( &
                sorted_cids(icell), nrows, ncols, ci, cj, is_valid)
            doms(ci, cj) = winner_dom
            saddles(sorted_cids(icell)) = nsaddles
            ! Go to the next cell with the same label
            icell = labels(icell)
        end do
    end subroutine process_plateau

    !> Computes topographic prominence, peak labels, and saddle 
    !! labels.
    !!
    !! Processes cells in descending elevation order using a
    !! topological sweep-plane algorithm. Local maxima initialise
    !! new peak domains, slopes extend existing peak domains, and
    !! saddles trigger peak domain merges where subordinate peaks
    !! receive their finalised prominence values.
    !!
    !! Outputs include the prominence height grid, 1-based peak 
    !! labels for summit cells, and 1-based saddle labels for key 
    !! saddle cells.
    !! Surviving global/regional maxima retain -1 to mark unknown/
    !! infinite prominence.
    subroutine compute_prominence( &
        z, sorted_cids, proms, peaks, saddles, nrows, ncols, nvalids, &
        offsets, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! DEM raster dimensions
        integer, intent(in) :: nvalids
            !! Number of valid DEM cells in sorted_cids
        real, intent(in) :: z(nrows, ncols)
            !! Input 2D DEM elevation grid
        integer(c_int32_t), intent(in) :: sorted_cids(nvalids)
            !! Valid cell linear indices sorted in ascending
            !! elevation order
        integer, intent(in) :: noffsets
            !! Number of neighbour connectivity directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of row and column offsets for neighbour
            !! connectivity
        ! Outputs
        real, intent(out) :: proms(nrows, ncols)
            !! Output topographic prominence height grid
        integer, intent(out) :: peaks(nrows, ncols)
            !! Output 1-based peak label grid
        integer, intent(out) :: saddles(nrows, ncols)
            !! Output 1-based saddle label grid
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: icell, jcell
            !! Loop indices traversing sorted cells
        integer :: ci, cj
            !! Row and column coordinates of current cell
        logical(kind=1) :: is_valid
            !! True if index conversion succeeded
        real :: slice_z
            !! Elevation of the current horizontal slice/plateau
        integer :: slice_start, slice_end
            !! Index bounds in 'sorted_cids' of cells with elevation
            !! equal to slice_z
        integer(c_int32_t) :: label
            !! Connected component label
        integer(c_int32_t), allocatable :: labels(:), label_heads(:)
            !! Linked-list 'next' pointers and bucket head indices
            !! per component
        integer :: ilabel, nlabels
            !! Component loop index and count of components at
            !! slice_z
        integer :: npeaks, nsaddles
            !! Total number of identified peaks and key saddles
        integer(c_int32_t), allocatable :: doms(:, :)
            !! 2D peak domain grid recording owning peak for each
            !! cell
        integer(c_int32_t), allocatable :: prnt_doms(:)
            !! Disjoint-set parent array tracking peak domain merges
        integer(c_int32_t), allocatable :: higher_doms(:)
            !! Buffer for adjacent higher domain IDs at a saddle
        integer(c_int32_t), allocatable :: copeaks(:)
            !! Singly-linked list tracking tied summits of identical
            !! elevation
        integer(c_int32_t), allocatable :: order_lookup(:)
            !! Inverse lookup mapping linear cell ID to position in
            !! 'sorted_cids'
        integer :: alloc_stat
            !! Allocation status indicator

        err_code = ERR_NO_ERROR
        allocate ( &
            labels(nvalids), label_heads(nvalids), &
            doms(nrows, ncols), prnt_doms(nvalids), &
            higher_doms(nvalids), &
            copeaks(nvalids), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        proms = 0
        peaks = 0
        saddles = 0
        npeaks = 0
        nsaddles = 0
        doms = 0
        copeaks = 0

        allocate (order_lookup(nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        order_lookup = 0
        do icell = 1, nvalids
            if (sorted_cids(icell) < 1 .or. &
                sorted_cids(icell) > nrows*ncols) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            prnt_doms(icell) = icell
            ! Build inverse lookup from linear ID in 'z' to position
            ! in 'sorted_cids'
            order_lookup(sorted_cids(icell)) = icell
        end do

        icell = nvalids
        do while (icell >= 1)
            call id2ij_checked( &
                sorted_cids(icell), nrows, ncols, ci, cj, is_valid)
            if (.not. is_valid) then
                err_code = ERR_INVALID_INPUT
                return
            end if
            slice_z = z(ci, cj)
            slice_end = icell
            slice_start = icell
            do while (icell >= 1)
                call id2ij_checked( &
                    sorted_cids(icell), nrows, ncols, ci, cj, is_valid)
                if (.not. is_valid) then
                    err_code = ERR_INVALID_INPUT
                    return
                end if
                if (z(ci, cj) /= slice_z) exit
                icell = icell - 1
            end do
            slice_start = icell + 1

            ! Group connected regions of the same elevation
            call label_plateaus( &
                sorted_cids(slice_start:slice_end), &
                labels(slice_start:slice_end), &
                nlabels, offsets, nrows, ncols, &
                slice_start, slice_end, order_lookup, err_code)
            if (err_code /= ERR_NO_ERROR) return

            label_heads(1:nlabels) = 0
            do jcell = slice_start, slice_end
                label = labels(jcell)
                ! Insert jcell at the front of this component's list
                labels(jcell) = label_heads(label)
                label_heads(label) = jcell
            end do

            ! Process each connected region one at a time
            do ilabel = 1, nlabels
                call process_plateau( &
                    z, sorted_cids, proms, &
                    peaks, saddles, npeaks, nsaddles, offsets, &
                    doms, label_heads, labels, ilabel, &
                    higher_doms, copeaks, prnt_doms, &
                    slice_z, err_code)
                if (err_code /= ERR_NO_ERROR) return
            end do
        end do
    end subroutine compute_prominence
end module terrain
