!!!
! Backend routines to assign flat areas in a digital elevation model
! (DEM) some synthetic elevation difference and therefore a gradient.
!
! The algorithms mainly follow
! [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009).
!
! Content of this file is mostly designed to be called by the Python
! frontend and not directly by the user.
!
! Last modified: 2026-08-05, En-Chi Lee (williameclee@gmail.com)
!!!

module drainage_flat_resolution
    use iso_c_binding, only: c_int8_t
    use utils, only: find_noflow_code, array2d_oob, mask2ij
    implicit none(type, external)
contains
    subroutine compute_syn_flowdir( &
        z, flats, dirs, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a synthetic elevation grid,
        !! using the provided flow direction codes and offsets.
        !!
        !! The flow directions are only computed for cells that are
        !! part of flats, as indicated by the  label grid. For each
        !! flat cell, the flow direction is assigned towards the
        !! neighbour with the lowest elevation within the same flat
        !! region. If no neighbour has a lower elevation, the cell
        !! is assigned the no-flow code.
        !!
        !! Note: This function is intended to be used for the
        !! synthetic terrain to resolve flats, which should be
        !! integer-typed.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: z(nrows, ncols)
            !! Synthetic elevation grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0
            !! for non-flat cells)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
        ! Outputs
        integer(c_int8_t), intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        ! Local variables
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to no-flow direction, to be
            !! determined from offsets and codes
        integer :: ci, cj, ni, nj
            !! (Cell-private) Rows/columns for current and neighbour
            !! cells
        integer :: iofs
            !! (Cell-private) Offset index for iterating through
            !! flow directions
        integer :: zmin
            !! (Cell-private) Minimum elevation among valid
            !! neighbours

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)
        dirs = noflow_code

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, iofs, zmin) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (flats(ci, cj) == 0) cycle

                zmin = z(ci, cj)

                do iofs = 1, noffsets
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    ! Skip if neighbour is different flat
                    if (flats(ni, nj) /= flats(ci, cj)) cycle
                    ! Check if neighbour has lower elevation
                    if (z(ni, nj) < zmin) then
                        zmin = z(ni, nj)
                        dirs(ci, cj) = codes(iofs)
                    end if
                end do
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine compute_syn_flowdir

    subroutine find_flat_edges( &
        z, dirs, valids, is_low_edge, is_high_edge, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds the cells on the edges of flat areas that drain to
        !! lower terrain (low edges) and those that are adjacent to
        !! higher terrain (high edges).
        !!
        !! From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009),
        !! Algorithm 3 (p. 133).
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
        ! Outputs
        logical(kind=1), intent(out) :: &
            is_low_edge(nrows, ncols), is_high_edge(nrows, ncols)
            !! Whether each cell is a 'low edge' or a 'high edge of a
            !! flat
        ! Local variables
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to no-flow direction, to be
            !! determined from offsets and codes
        integer :: ci, cj, ni, nj
            !! (Cell-private) Rows/columns for current and neighbour
            !! cells
        integer :: iofs
            !! (Cell-private) Offset index for iterating through
            !! flow directions

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        is_low_edge = .false.
        is_high_edge = .false.

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, iofs) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle

                do iofs = 1, noffsets
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    ! Skip if neighbour is not valid
                    if (.not. valids(ni, nj)) cycle
                    ! Check for low edge
                    if (dirs(ci, cj) /= noflow_code .and. &
                        dirs(ni, nj) == noflow_code .and. &
                        z(ci, cj) == z(ni, nj)) then
                        is_low_edge(ci, cj) = .true.
                        exit
                    end if
                    ! Check for high edge
                    if (dirs(ci, cj) == noflow_code .and. &
                        z(ci, cj) < z(ni, nj)) then
                        is_high_edge(ci, cj) = .true.
                        exit
                    end if
                end do
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine find_flat_edges

    pure subroutine label_flats( &
        z, seeds, valids, flats, nrows, ncols, &
        offsets, noffsets, err_code)
        !! Labels connected flat regions in the elevation grid,
        !! using a flood-fill algorithm starting from the provided
        !! seed cells.
        !!
        !! Only valid cells (as indicated by the valids mask) will
        !! be considered for labelling. Each flat region will be
        !! assigned a unique integer label in the output  grid,
        !! while non-flat cells will be assigned 0.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: seeds(nrows, ncols)
            !! Seed mask indicating starting points for labelling
            !! flat regions
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0
            !! for non-flat cells)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A high-edge seed does not belong to a labelled
            !!     flat
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Flat-flooding buffer capacity was exceeded
        ! Local variables
        integer :: iflat
            !! Index of the current flat region being labeled
            !! (!= issed because same flat can have multiple seeds)
        integer, allocatable :: seed_ijs(:, :)
            !! List of (i, j) indices for seed cells
        integer :: iseed, nseeds
            !! Index and total number of seed cells ('seed_ijs')
        integer, allocatable :: flat_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be
            !! filled in the current flat region
        integer :: ifill, nfills
            !! Index and total number of cells in the current flat
            !! region being filled ('flat_ijs')
        integer :: si, sj, ci, cj, ni, nj
            !! Rows/columns for seed, current and neighbour cells
        real :: sz
            !! Elevation of the current flat region being labeled
        integer :: iofs
            !! Index for iterating through offsets

        err_code = 0
        allocate (flat_ijs(2, nrows*ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (seed_ijs(2, nrows*ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        ! Convert seed mask to list of (i, j) indices
        call mask2ij( &
            seeds, seed_ijs, size(seed_ijs, dim=2), nseeds, err_code)
        if (err_code /= 0) return

        flats = 0
        iflat = 1
        iseed = 1
        ! Loop over seed cells to label flats using a flood-fill
        ! algorithm
        do iseed = 1, nseeds
            si = seed_ijs(1, iseed)
            sj = seed_ijs(2, iseed)

            ! Skip if not valid
            if (.not. valids(si, sj)) cycle
            ! Skip if already labeled
            if (flats(si, sj) /= 0) cycle

            sz = z(si, sj)

            ! Reset buffer
            ifill = 1
            nfills = 1
            flat_ijs(:, ifill) = [si, sj]
            flats(si, sj) = iflat

            do while (ifill <= nfills)
                ci = flat_ijs(1, ifill)
                cj = flat_ijs(2, ifill)
                ifill = ifill + 1

                ! Loop over offsets to find connected flat cells
                do iofs = 1, noffsets
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    ! Skip if not valid
                    if (.not. valids(ni, nj)) cycle
                    ! Skip if already labeled
                    if (flats(ni, nj) /= 0) cycle
                    ! Skip if not the same flat (i.e. different elevation)
                    if (z(ni, nj) /= sz) cycle
                    ! Add to tofill buffer
                    nfills = nfills + 1
                    if (nfills > size(flat_ijs, dim=2)) then
                        err_code = 3
                        return
                    end if
                    flat_ijs(:, nfills) = [ni, nj]
                    flats(ni, nj) = iflat
                end do

            end do

            iflat = iflat + 1
        end do
        deallocate (flat_ijs)
        deallocate (seed_ijs)
    end subroutine label_flats

    pure subroutine create_pushing_syn_grad( &
        z, flats, nrows, ncols, &
        high_edges, offsets, noffsets, err_code)
        !! Produces a synthetic elevation that decreases away from
        !! 'high edges' of flats.
        !!
        !! Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009),
        !! Algorithm 5 (p. 133--134).
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0
            !! for non-flat cells)
        logical(kind=1), intent(in) :: high_edges(nrows, ncols)
            !! Mask indicating which cells are 'high edges'
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: z(nrows, ncols)
            !! Synthetic elevation grid that has the down gradient
            !! flow away from high edges
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: High-edge queue capacity was exceeded or an
            !!     index was out of bounds
        ! Local variables
        integer :: nflats
            !! Number of unique flat labels (excluding 0 for
            !! non-flat cells)
        integer :: dist
            !! Current distance from high edges, used to assign
            !! synthetic elevation values
        integer, allocatable :: maxdist(:)
            !! Maximum synthetic elevation value assigned to each
            !! flat region, used to adjust final z values to ensure
            !! they flow away from high edges
        integer :: iedge, nedges, layer_end
            !! Index for iterating through high edge cells and total
            !! number of high edge cells in the queue
            !! As the algorithm proceeds, new cells will be added to
            !! the queue and nedges will be updated accordingly
        integer :: iofs
            !! Index for iterating through offsets
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical(kind=1), allocatable :: queued(:, :)
            !! Mask to track which cells have already been added to
            !! the queue, to avoid adding the same cell multiple
            !! times
        integer, allocatable :: high_edge_ijs(:, :)
            !! List of (i, j) indices for high edge cells to be
            !! processed in the algorithm, used as a queue for
            !! breadth-first search
        integer :: max_queue_size
            !! Maximum size of the queue buffer for high edge cells

        err_code = 0
        ! Each labelled flat cell is queued at most once
        ! Track breadth-first layers with an index instead of
        ! storing layer markers in the queue.
        max_queue_size = count(flats /= 0)
        z = 0
        if (max_queue_size == 0) return
        allocate (high_edge_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if

        nedges = 0
        call mask2ij( &
            high_edges, high_edge_ijs, size(high_edge_ijs, dim=2), nedges, &
            err_code)
        if (err_code /= 0) return
        if (nedges == 0) then
            ! No high edges found, set z to zero and exit
            deallocate (high_edge_ijs)
            return
        end if

        nflats = maxval(flats)
        allocate (maxdist(nflats), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        maxdist = 0

        allocate (queued(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        queued = .false.
        ! Mark initial seeds as queued
        do iedge = 1, nedges
            ci = high_edge_ijs(1, iedge)
            cj = high_edge_ijs(2, iedge)
            queued(ci, cj) = .true.
        end do
        ! Loop through all high_edges to find cells flowing away
        ! from flats
        ! After this the first loop, z values decreases towards high
        ! edges (opposite of desired)
        dist = 1
        iedge = 1
        layer_end = nedges
        do while (iedge <= nedges)
            ci = high_edge_ijs(1, iedge)
            cj = high_edge_ijs(2, iedge)
            iedge = iedge + 1

            if (array2d_oob(ci, cj, nrows, ncols)) then
                err_code = 3
                return
            else if (flats(ci, cj) == 0) then
                err_code = 1
                return
            end if

            z(ci, cj) = dist
            maxdist(flats(ci, cj)) = dist

            ! Loop over offsets to find contributing cells
            do iofs = 1, noffsets
                ! Skip self
                if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle

                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)

                ! Check bounds
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                ! Skip if already queued
                if (queued(ni, nj)) cycle
                ! Skip if not a flat
                if (flats(ni, nj) == 0) cycle
                ! Skip if already processed
                if (z(ni, nj) > 0) cycle
                ! Skip if different flat
                if (flats(ni, nj) /= flats(ci, cj)) cycle
                ! Update queue
                nedges = nedges + 1
                if (nedges > max_queue_size) then
                    err_code = 3
                    return
                end if
                high_edge_ijs(:, nedges) = [ni, nj]
                queued(ni, nj) = .true.
            end do

            if (iedge > layer_end) then
                dist = dist + 1
                layer_end = nedges
            end if
        end do
        deallocate (high_edge_ijs)
        deallocate (queued)

        ! Adjust z values within flats to ensure they flow away from
        ! high edges
        do concurrent(ci=1:nrows, cj=1:ncols, flats(ci, cj) /= 0)
            z(ci, cj) = maxdist(flats(ci, cj)) - z(ci, cj) + 1
        end do
        deallocate (maxdist)
    end subroutine create_pushing_syn_grad

    pure subroutine create_pulling_syn_grad( &
        z, flats, nrows, ncols, &
        low_edges, offsets, noffsets, err_code)
        !! Produces a synthetic elevation that drains towards 'low
        !! edges' of flats.
        !!
        !! Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009),
        !! Algorithm 6 (p. 134).
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0
            !! for non-flat cells)
        logical(kind=1), intent(in) :: low_edges(nrows, ncols)
            !! Mask indicating which cells are 'low edges'
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: z(nrows, ncols)
            !! Synthetic elevation grid that drains towards low edges
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Low-edge queue capacity was exceeded or an
            !!     index was out of bounds
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer :: iedge, nedges, layer_end
            !! Index for iterating through low edge cells and total
            !! number of low edge cells in the queue
        integer :: dist
            !! Current distance from low edges, used to assign
            !! synthetic elevation values
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical(kind=1), allocatable :: queued(:, :)
            !! Mask to track which cells have already been added to
            !! the queue, to avoid adding the same cell multiple
            !! times
        integer, allocatable :: low_edges_ijs(:, :)
            !! List of (i, j) indices for low edge cells to be
            !! processed in the algorithm, used as a queue for
            !! breadth-first search
        integer :: max_queue_size
            !! Maximum size of the queue buffer for low edge cells

        err_code = 0
        ! Each labelled flat cell is queued at most once
        ! Track breadth-first layers with an index instead of
        ! storing layer markers in the queue
        max_queue_size = count(flats /= 0)
        z = 0
        if (max_queue_size == 0) return
        allocate (low_edges_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        call mask2ij( &
            low_edges, low_edges_ijs, size(low_edges_ijs, dim=2), nedges, &
            err_code)
        if (err_code /= 0) return
        if (nedges == 0) then
            deallocate (low_edges_ijs)
            return
        end if
        allocate (queued(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        queued = .false.

        ! Mark initial seeds as queued
        do iedge = 1, nedges
            ci = low_edges_ijs(1, iedge)
            cj = low_edges_ijs(2, iedge)
            queued(ci, cj) = .true.
        end do

        ! Loop through all low_edges to find cells flowing into flats
        iedge = 1
        dist = 1
        layer_end = nedges
        do while (iedge <= nedges)
            ci = low_edges_ijs(1, iedge)
            cj = low_edges_ijs(2, iedge)
            iedge = iedge + 1

            if (array2d_oob(ci, cj, nrows, ncols)) then
                err_code = 3
                return
            end if

            ! Queueing should guarantee we only visit each cell once
            z(ci, cj) = dist

            ! Loope over offsets to find contributing cells
            do iofs = 1, noffsets
                ! Skip self
                if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle

                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)

                ! Check bounds
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                ! Check if already queued
                if (queued(ni, nj)) cycle
                ! Skip if not a flat
                if (flats(ni, nj) == 0) cycle
                ! Skip if already processed
                if (z(ni, nj) > 0) cycle
                ! Skip if different flat
                if (flats(ni, nj) /= flats(ci, cj)) cycle

                ! Update queue
                nedges = nedges + 1
                if (nedges > max_queue_size) then
                    err_code = 3
                    return
                end if
                low_edges_ijs(:, nedges) = [ni, nj]
                queued(ni, nj) = .true.
            end do

            if (iedge > layer_end) then
                dist = dist + 1
                layer_end = nedges
            end if
        end do
        deallocate (queued)
        deallocate (low_edges_ijs)
    end subroutine create_pulling_syn_grad
end module drainage_flat_resolution
