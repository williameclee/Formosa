!!!
! Last modified
!   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
!     - Rename flowdir functions to be more descriptive
!   2026-06-09, En-Chi Lee (williameclee@gmail.com)
!     - Small refactors and documentation cleanup
!     - Renamed function: 'compute_masked_flowdir' -> 'compute_synthetic_flowdir'
!     - Added valids argument to 'label_flats' function
!   2026-06-10, En-Chi Lee (williameclee@gmail.com)
!     - Small refactors and documentation cleanup
!   2026-06-11, En-Chi Lee (williameclee@gmail.com)
!     - Added precomputed 'dist_lookup' for L1 distance in 'compute_dist2source_l1'
!     - Standardised variable, argument, and function names
!   2026-07-01, En-Chi Lee (williameclee@gmail.com)
!     - Fixed Strahler order algorithm
!     - Optimised confluence lookup algorithm
!     - Changed index array shape to optimise cache locality
!     - Allowed specifying validity mask in 'count_indegree'
!   2026-07-08, En-Chi Lee (williameclee@gmail.com)
!     - Moved 'mask2ij' to separate 'utils' module
!   2026-07-09, En-Chi Lee (williameclee@gmail.com)
!     - Fixed OpenMP data race in 'count_indegree'
!   2026-07-14, En-Chi Lee (williameclee@gmail.com)
!     - Splitted 'flowdir_f' into submodules
!   2026-08-03, En-Chi Lee (williameclee@gmail.com)
!     - Implemented 'find_acyclic_flowdirs'
!     - Explicitly handled Python uint8 -> FORTRAN INTEGER*1 conversion/interpretation in 'fill_offset_lookup'
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Added allocation error monitoring and moved error handling to Python
!!!

module flowdir_raster
    use omp_lib
    use utils
    use distances
    implicit none
contains
    subroutine compute_flowdir_simple( &
        z, valids, dirs, is_flat, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a given elevation grid, using
        !! the provided flow direction codes and offsets.
        !! Also identifies flat cells where no flow direction can be
        !! assigned.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer*1, intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(out) :: is_flat(nrows, ncols)
            !! Mask indicating which cells are part of flats (i.e. direction is no-flow)
        ! Local variables
        integer*1 :: noflow_code
            !! Code corresponding to no-flow direction, to be determined from offsets and codes
        integer :: ci, cj, ni, nj
            !! (Cell-private) Rows/columns for current and neighbour cells
        integer :: iofs
            !! (Cell-private) Offset index for iterating through flow directions
        real :: zmin
            !! (Cell-private) Minimum elevation among valid neighbours

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)
        dirs = noflow_code
        is_flat = .false.

        !! Main loop to compute flow directions
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, iofs, zmin) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle

                zmin = z(ci, cj)

                do iofs = 1, noffsets
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    ! Check if neighbour is part of the same flat
                    if (.not. valids(ni, nj)) cycle
                    ! Check if neighbour has lower elevation
                    if (z(ni, nj) < zmin) then
                        zmin = z(ni, nj)
                        dirs(ci, cj) = codes(iofs)
                    end if
                end do
                if (dirs(ci, cj) == noflow_code) then
                    is_flat(ci, cj) = .true.
                end if
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine compute_flowdir_simple

    subroutine compute_syn_flowdir( &
        z, flats, dirs, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a synthetic elevation grid,
        !! using the provided flow direction codes and offsets.
        !! The flow directions are only computed for cells that are part
        !! of flats, as indicated by the  label grid. For each flat
        !! cell, the flow direction is assigned towards the neighbour
        !! with the lowest elevation within the same flat region. If no
        !! neighbour has a lower elevation, the cell is assigned the
        !! no-flow code.
        !! Note: This function is intended to be used for the synthetic
        !! terrain to resolve flats, which should be integer-typed.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: z(nrows, ncols)
            !! Synthetic elevation grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer*1, intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        ! Local variables
        integer*1 :: noflow_code
            !! Code corresponding to no-flow direction, to be determined from offsets and codes
        integer :: ci, cj, ni, nj
            !! (Cell-private) Rows/columns for current and neighbour cells
        integer :: iofs
            !! (Cell-private) Offset index for iterating through flow directions
        integer :: zmin
            !! (Cell-private) Minimum elevation among valid neighbours

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
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
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
        !! From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 3 (p. 133).
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical*1, intent(out) :: is_low_edge(nrows, ncols), is_high_edge(nrows, ncols)
            !! Whether each cell is a 'low edge' or a 'high edge of a flat.
        ! Local variables
        integer*1 :: noflow_code
            !! Code corresponding to no-flow direction, to be determined from offsets and codes
        integer :: ci, cj, ni, nj
            !! (Cell-private) Rows/columns for current and neighbour cells
        integer :: iofs
            !! (Cell-private) Offset index for iterating through flow directions

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
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    ! Skip if neighbour is not valid
                    if (.not. valids(ni, nj)) cycle
                    ! Check for low edge
                    if (dirs(ci, cj) /= noflow_code .and. dirs(ni, nj) == noflow_code .and. z(ci, cj) == z(ni, nj)) then
                        is_low_edge(ci, cj) = .true.
                        exit
                    end if
                    ! Check for high edge
                    if (dirs(ci, cj) == noflow_code .and. z(ci, cj) < z(ni, nj)) then
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
        !! Labels connected flat regions in the elevation grid, using a
        !! flood-fill algorithm starting from the provided seed cells.
        !! Only valid cells (as indicated by the valids mask) will be
        !! considered for labelling. Each flat region will be assigned a
        !! unique integer label in the output  grid, while non-flat
        !! cells will be assigned 0.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical*1, intent(in) :: seeds(nrows, ncols)
            !! Seed mask indicating starting points for labelling flat regions
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A high-edge seed does not belong to a labelled flat
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Flat-flooding buffer capacity was exceeded
        ! Local variables
        integer :: iflat
            !! Index of the current flat region being labeled (!= issed because same flat can have multiple seeds)
        integer, allocatable :: seed_ijs(:, :)
            !! List of (i, j) indices for seed cells
        integer :: iseed, nseeds
            !! Index and total number of seed cells ('seed_ijs')
        integer, allocatable :: flat_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be filled in the current flat region
        integer :: ifill, nfills
            !! Index and total number of cells in the current flat region being filled ('flat_ijs')
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
        call mask2ij(seeds, seed_ijs, size(seed_ijs, dim=2), nseeds, err_code)
        if (err_code /= 0) return

        flats = 0
        iflat = 1
        iseed = 1
        ! Loop over seed cells to label flats using a flood-fill algorithm
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
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
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
        !! Produces a synthetic elevation that decreases away from 'high
        !! edges' of flats.
        !! Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 5 (p. 133--134).
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
        logical*1, intent(in) :: high_edges(nrows, ncols)
            !! Mask indicating which cells are 'high edges'
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: z(nrows, ncols)
            !! Synthetic elevation grid that has the down gradient flow away from high edges
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: High-edge queue capacity was exceeded or an index was out of bounds
        ! Local variables
        integer :: nflats
            !! Number of unique flat labels (excluding 0 for non-flat cells)
        integer :: dist
            !! Current distance from high edges, used to assign synthetic elevation values
        integer, allocatable :: maxdist(:)
            !! Maximum synthetic elevation value assigned to each flat region, used to adjust final z values to ensure they flow away from high edges
        integer :: iedge, nedges, layer_end
            !! Index for iterating through high edge cells and total number of high edge cells in the queue
            !! As the algorithm proceeds, new cells will be added to the queue and nedges will be updated accordingly
        integer :: iofs
            !! Index for iterating through offsets
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical*1, allocatable :: queued(:, :)
            !! Mask to track which cells have already been added to the queue, to avoid adding the same cell multiple times
        integer, allocatable :: high_edge_ijs(:, :)
            !! List of (i, j) indices for high edge cells to be processed in the algorithm, used as a queue for breadth-first search
        integer :: max_queue_size
            !! Maximum size of the queue buffer for high edge cells

        err_code = 0
        ! Each labelled flat cell is queued at most once. Track breadth-first
        ! layers with an index instead of storing layer markers in the queue.
        max_queue_size = count(flats /= 0)
        z = 0
        if (max_queue_size == 0) return
        allocate (high_edge_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if

        nedges = 0
        call mask2ij(high_edges, high_edge_ijs, size(high_edge_ijs, dim=2), nedges, err_code)
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
        ! Loop through all high_edges to find cells flowing away from flats
        ! After this the first loop, z values decreases towards high edges (opposite of desired)
        dist = 1
        iedge = 1
        layer_end = nedges
        do while (iedge <= nedges)
            ci = high_edge_ijs(1, iedge)
            cj = high_edge_ijs(2, iedge)
            iedge = iedge + 1

            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
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
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
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

        ! Adjust z values within flats to ensure they flow away from high edges
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
        !! Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 6 (p. 134).
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
        logical*1, intent(in) :: low_edges(nrows, ncols)
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
            !!   - 3: Low-edge queue capacity was exceeded or an index was out of bounds
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer :: iedge, nedges, layer_end
            !! Index for iterating through low edge cells and total number of low edge cells in the queue
        integer :: dist
            !! Current distance from low edges, used to assign synthetic elevation values
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical*1, allocatable :: queued(:, :)
            !! Mask to track which cells have already been added to the queue, to avoid adding the same cell multiple times
        integer, allocatable :: low_edges_ijs(:, :)
            !! List of (i, j) indices for low edge cells to be processed in the algorithm, used as a queue for breadth-first search
        integer :: max_queue_size
            !! Maximum size of the queue buffer for low edge cells

        err_code = 0
        ! Each labelled flat cell is queued at most once. Track breadth-first
        ! layers with an index instead of storing layer markers in the queue.
        max_queue_size = count(flats /= 0)
        z = 0
        if (max_queue_size == 0) return
        allocate (low_edges_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        call mask2ij(low_edges, low_edges_ijs, size(low_edges_ijs, dim=2), nedges, err_code)
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

            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
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
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
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

    subroutine count_indegree( &
        dirs, valids, indegs, nrows, ncols, &
        offsets, codes, noffsets)
        !! Computes the number of upstream cells (indegs) for each cell
        !! in a flow direction grid.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer*1, intent(out) :: indegs(nrows, ncols)
            !! Grid of indegree values, i.e. number of upstream cells that flow into each cell
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells

        indegs = 0

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, iofs) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle

                ! Loop over offsets to find neighbours flowing into current cell
                do iofs = 1, noffsets
                    ! Upstream neighbour indices
                    ni = ci - offsets(iofs, 1)
                    nj = cj - offsets(iofs, 2)
                    ! Check bounds
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    ! Check if neighbour is valid
                    if (.not. valids(ni, nj)) cycle
                    ! Skip self-loops
                    if (ni == ci .and. nj == cj) cycle
                    ! Check if neighbour flows into current cell
                    if (dirs(ni, nj) == codes(iofs)) then
                        indegs(ci, cj) = indegs(ci, cj) + int(1, kind=1)
                    end if
                end do
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine count_indegree

    subroutine compute_flow_accumulation( &
        dirs, valids, areas, indegs, accumulations, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes flow accumulation for each cell in a flow direction grid.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        real, intent(in) :: areas(nrows, ncols)
            !! Area of each cell, used as the initial accumulation value for each cell
        integer*1, intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell.
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: accumulations(nrows, ncols)
            !! Grid of flow accumulation values, i.e. total area flowing into each cell
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Flooding queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        integer, allocatable :: flood_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the flooding buffer ('flood_ijs')
        logical*1, allocatable :: flood_seeds(:, :)
            !! Mask to identify initial seed cells for the flooding algorithm (valid cells with zero in-degrees)

        ! Create lookup tables for offsets
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill the tofill buffer with all valid cells with zero in-degrees
        max_queue_size = nrows*ncols
        allocate (flood_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (flood_seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        flood_seeds = valids .and. (indegs == 0)
        call mask2ij(flood_seeds, flood_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (flood_seeds)

        err_code = 0
        accumulations = areas
        itofill = 1
        do while (itofill <= ntofills)
            ci = flood_ijs(1, itofill)
            cj = flood_ijs(2, itofill)
            itofill = itofill + 1

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update accumulation of downstream cell
            accumulations(ni, nj) = accumulations(ni, nj) + accumulations(ci, cj)
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to flooding buffer
            if (indegs(ni, nj) > 0) cycle
            ntofills = ntofills + 1
            if (ntofills > max_queue_size) then
                err_code = 3
                return
            end if
            flood_ijs(:, ntofills) = [ni, nj]
        end do
        deallocate (offset_lookup)
        deallocate (flood_ijs)
    end subroutine compute_flow_accumulation

    subroutine compute_dist2source_l1( &
        dirs, valids, indegs, dists, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes the distance to the nearest source cell (cell with
        !! zero indegree) for each cell in a flow direction grid, using
        !! a breadth-first search starting from source cells.
        !! The distance in measured in the number of cells along the
        !! flow path (i.e. integer-typed L1 distance).
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer*1, intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer, intent(out) :: dists(nrows, ncols)
            !! Grid of distances to the nearest source cell (cell with zero indegree).
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Source-distance queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        ! integer :: step_dist
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical*1, allocatable :: tofill_seeds(:, :)
            !! Mask to identify initial seed cells for the flooding algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the flooding buffer ('tofill_ijs')

        ! Create lookup tables for offsets
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (tofill_seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        tofill_seeds = valids .and. (indegs == 0)
        call mask2ij(tofill_seeds, tofill_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (tofill_seeds)

        !! Main loop to fill distances using a breadth-first search starting from source cells
        err_code = 0
        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(1, itofill)
            cj = tofill_ijs(2, itofill)
            itofill = itofill + 1

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            dists(ni, nj) = max(dists(ci, cj), dists(ci, cj) + l1dist_xy(ni, nj, ci, cj))
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > max_queue_size) then
                    err_code = 3
                    return
                end if
                tofill_ijs(:, ntofills) = [ni, nj]
            end if
        end do
        deallocate (offset_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_dist2source_l1

    subroutine compute_dist2source( &
        dirs, valids, x, y, indegs, dists, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes the distance downstream along flow directions for
        !! each cell in the flow direction grid.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        integer*1, intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: dists(nrows, ncols)
            !! Grid of distances to the nearest source cell
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Source-distance queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the flooding algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the flooding buffer ('tofill_ijs')

        ! Create lookup tables for offsets
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        seeds = valids .and. (indegs == 0)
        call mask2ij(seeds, tofill_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        !! Main loop to fill distances using a breadth-first search starting from source cells
        err_code = 0
        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(1, itofill)
            cj = tofill_ijs(2, itofill)
            itofill = itofill + 1

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            dists(ni, nj) = max(dists(ci, cj), dists(ci, cj) + l2dist_xy(x(ni, nj), y(ni, nj), x(ci, cj), y(ci, cj)))
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > max_queue_size) then
                    err_code = 3
                    return
                end if
                tofill_ijs(:, ntofills) = [ni, nj]
            end if
        end do
        deallocate (offset_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_dist2source

    subroutine compute_dist2sink( &
        dists, dirs, x, y, valids, nrows, ncols, offsets, codes, noffsets, err_code)
        !! Computes the distance upstream along flow directions for each cell in the flow direction grid.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: dists(nrows, ncols)
            !! Grid of distances to the downstream sink
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Sink-distance queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer*1 :: noflow_code
            !! Code corresponding to noflow direction, used to identify sink cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream cells
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid cells with noflow direction)
        integer, allocatable :: seed_ijs(:, :)
            !! Buffer for storing (i, j) indices of seed cells to be processed in the algorithm
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the breadth-first search from sink cells
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed ('seed_ijs' and 'tofill_ijs')
        integer :: alloc_stat
            !! Per-thread allocation status code
        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        dists = -1

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        seeds = valids .and. (dirs == noflow_code)
        call mask2ij(seeds, seed_ijs, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs, alloc_stat)
        allocate (tofill_ijs(2, max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            si = seed_ijs(1, iseed)
            sj = seed_ijs(2, iseed)

            ! Loop through buffer
            nfills = 1
            ifill = 1
            dists(si, sj) = 0.0
            tofill_ijs(:, 1) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(1, ifill)
                cj = tofill_ijs(2, ifill)
                ifill = ifill + 1

                ! Loop over offsets to find contributing cells
                do iofs = 1, noffsets
                    ! Skip self
                    if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle
                    ui = ci - offsets(iofs, 1)
                    uj = cj - offsets(iofs, 2)

                    ! Check bounds
                    if (ui < 1 .or. ui > nrows .or. uj < 1 .or. uj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ui, uj)) cycle
                    ! Check if already assigned
                    if (dists(ui, uj) >= 0) cycle
                    ! Check if flows into current cell
                    if (dirs(ui, uj) /= codes(iofs)) cycle

                    ! Add to buffer
                    nfills = nfills + 1
                    if (nfills > max_queue_size) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ijs(:, nfills) = [ui, uj]
                    ! Compute distance
                    dists(ui, uj) = dists(ci, cj) &
                                    + l2dist_xy(x(ui, uj), y(ui, uj), x(ci, cj), y(ci, cj))
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ijs)) deallocate (tofill_ijs)
        !$omp END PARALLEL
        deallocate (seed_ijs)
    end subroutine compute_dist2sink

    subroutine compute_flow_strahler_order( &
        dirs, valids, indegs, orders, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer*1, intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer*2, intent(out) :: orders(nrows, ncols)
            !! Grid of Strahler stream order values for each cell
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Strahler traversal queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer :: iofs
            !! Index for iterating through offsets
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj, ui, uj
            !! Rows/columns for current and neighbour downstream and upstream cells
        integer*2 :: max_uorder
            !! Maximum Strahler stream order value of a cell's upstream neighbours
        logical*1 :: increase_order
            !! Whether the current cell's order should be increased
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the breadth-first search from source cells
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed ('tofill_ijs')

        ! Create lookup tables for offsets
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        seeds = valids .and. (indegs == 0)
        err_code = 0
        orders = merge(int(1, kind=2), int(0, kind=2), seeds)
        call mask2ij(seeds, tofill_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        itofill = 1

        ! Push all the seeds' downstream cells to the queue
        do while (itofill <= ntofills)
            ci = tofill_ijs(1, itofill)
            cj = tofill_ijs(2, itofill)
            itofill = itofill + 1

            if (orders(ci, cj) == 0) then

                ! Check upstream cells to assign the current ones' order
                max_uorder = 0
                increase_order = .false.
                do iofs = 1, noffsets
                    ui = ci - offsets(iofs, 1)
                    uj = cj - offsets(iofs, 2)

                    ! Check bounds
                    if (ui < 1 .or. ui > nrows .or. uj < 1 .or. uj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ui, uj)) cycle
                    ! Check it actually flows into the current cell
                    if (dirs(ui, uj) /= codes(iofs)) cycle
                    ! Check not a self-loop
                    if (ui == ci .and. uj == cj) cycle

                    if (orders(ui, uj) > max_uorder) then
                        max_uorder = orders(ui, uj)
                        increase_order = .false.
                    else if (orders(ui, uj) == max_uorder) then
                        increase_order = .true.
                    end if
                end do

                if (increase_order) then
                    orders(ci, cj) = max_uorder + int(1, kind=2)
                else
                    orders(ci, cj) = max_uorder
                end if
            end if

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not seed or already processed
            if (indegs(ni, nj) == 0) cycle

            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) > 0) cycle

            ntofills = ntofills + 1
            if (ntofills > max_queue_size) then
                err_code = 3
                return
            end if
            tofill_ijs(:, ntofills) = [ni, nj]

        end do
        deallocate (offset_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_flow_strahler_order

    subroutine label_watersheds( &
        labels, dirs, valids, nrows, ncols, offsets, codes, noffsets, err_code)
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer, intent(out) :: labels(nrows, ncols)
            !! Grid of watershed labels, where cells with the same label belong to the same watershed.
            !! Cells with no-data or that do not flow into any watershed should have a label of 0.
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Watershed traversal queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer*1 :: noflow_code
            !! Code corresponding to noflow direction, used to identify seed cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream indices
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid cells with noflow direction)
        integer, allocatable :: seed_ijs(:, :), tofill_ijs(:, :)
            !! Buffers for storing (i, j) indices of seed cells and cells to be processed in the breadth-first search from seed cells
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed ('seed_ijs' and 'tofill_ijs')
        integer :: alloc_stat
            !! Per-thread allocation status code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        labels = 0

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        allocate (seeds(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        seeds = valids .and. (dirs == noflow_code)
        call mask2ij(seeds, seed_ijs, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs, alloc_stat)
        allocate (tofill_ijs(2, max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            si = seed_ijs(1, iseed)
            sj = seed_ijs(2, iseed)

            ! Loop through buffer
            nfills = 1
            ifill = 1
            labels(si, sj) = iseed
            tofill_ijs(:, 1) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(1, ifill)
                cj = tofill_ijs(2, ifill)
                ifill = ifill + 1

                ! Loop over offsets to find contributing cells
                do iofs = 1, noffsets
                    ! Skip self
                    if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle
                    ui = ci - offsets(iofs, 1)
                    uj = cj - offsets(iofs, 2)

                    ! Check bounds
                    if (ui < 1 .or. ui > nrows .or. uj < 1 .or. uj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ui, uj)) cycle
                    ! Check if already assigned
                    if (labels(ui, uj) > 0) cycle
                    ! Check if flows into current cell
                    if (dirs(ui, uj) /= codes(iofs)) cycle

                    ! Add to buffer
                    nfills = nfills + 1
                    if (nfills > max_queue_size) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ijs(:, nfills) = [ui, uj]
                    ! Compute distance
                    labels(ui, uj) = labels(ci, cj)
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ijs)) deallocate (tofill_ijs)
        !$omp END PARALLEL
    end subroutine label_watersheds

    subroutine flood_upstream( &
        flooded, dirs, seeds, valids, nrows, ncols, offsets, codes, noffsets, err_code)
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols), seeds(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data) and seed mask (true for seed cells, false for non-seed cells)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical*1, intent(out) :: flooded(nrows, ncols)
            !! Mask indicating which cells are flooded (true for flooded cells, false for non-flooded cells)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Upstream-flooding queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer*1 :: noflow_code
            !! Code corresponding to noflow direction, used to identify seed cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream indices
        integer, allocatable :: seed_ijs(:, :), tofill_ijs(:, :)
            !! Buffers for storing (i, j) indices of seed cells and cells to be processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed ('seed_ijs' and 'tofill_ijs')
        integer :: alloc_stat
            !! Per-thread allocation status code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        flooded = .false.

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(2, max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        call mask2ij(seeds, seed_ijs, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs, alloc_stat)
        allocate (tofill_ijs(2, max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            si = seed_ijs(1, iseed)
            sj = seed_ijs(2, iseed)

            ! Check if is valid
            if (.not. valids(si, sj)) cycle

            ! Loop through buffer
            nfills = 1
            ifill = 1
            flooded(si, sj) = .true.
            tofill_ijs(:, 1) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(1, ifill)
                cj = tofill_ijs(2, ifill)
                ifill = ifill + 1

                ! Loop over offsets to find contributing cells
                do iofs = 1, noffsets
                    ! Skip self
                    if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle
                    ui = ci - offsets(iofs, 1)
                    uj = cj - offsets(iofs, 2)

                    ! Check bounds
                    if (ui < 1 .or. ui > nrows .or. uj < 1 .or. uj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ui, uj)) cycle
                    ! Check if already assigned
                    if (flooded(ui, uj)) cycle
                    ! Check if flows into current cell
                    if (dirs(ui, uj) /= codes(iofs)) cycle

                    ! Add to buffer
                    nfills = nfills + 1
                    if (nfills > max_queue_size) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ijs(:, nfills) = [ui, uj]
                    ! Compute distance
                    flooded(ui, uj) = .true.
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ijs)) deallocate (tofill_ijs)
        !$omp END PARALLEL
        deallocate (seed_ijs)
    end subroutine flood_upstream

    subroutine find_acyclic_flowdirs( &
        dirs, indegs, valids, nrows, ncols, offsets, codes, noffsets, acyclics, err_code)
        !! Identifies valid cells that are not part of a directed flow cycle.
        !! Uses Kahn's algorithm to traverse cells from zero-indegree seeds,
        !! successively removing their outgoing edges. Valid cells not reached
        !! by this traversal belong to a directed cycle and remain false in
        !! 'acyclics'.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        integer*1, intent(in) :: indegs(nrows, ncols)
            !! Indegree grid for the valid flow field
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical*1, intent(out) :: acyclics(nrows, ncols)
            !! Mask indicating valid cells removed by Kahn's algorithm
            !! (true for acyclic cells, false otherwise)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Traversal workspace allocation failed
            !!   - 3: Acyclic traversal queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer*1, allocatable :: rem_indegs(:, :)
            !! Remaining indegrees after removing edges from processed cells
        logical*1, allocatable :: seeds(:, :)
            !! Mask of valid zero-indegree cells used to initialise the queue
        integer, allocatable :: seed_ijs(:, :)
            !! Queue of (i, j) indices awaiting processing
        integer :: alloc_stat
            !! Allocation status code
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and downstream cells
        integer :: iseed, nseeds
            !! Current queue position and final occupied queue position

        err_code = 0

        allocate (offset_lookup(0:255, 2), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            return
        end if

        allocate (rem_indegs(nrows, ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            return
        end if
        allocate (seeds(nrows, ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            deallocate (rem_indegs)
            return
        end if

        allocate (seed_ijs(2, nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            deallocate (rem_indegs)
            deallocate (seeds)
            return
        end if

        seeds = valids .and. (indegs == 0)
        offset_lookup = fill_offset_lookup(offsets, codes)
        call mask2ij(seeds, seed_ijs, size(seed_ijs, dim=2), nseeds, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        rem_indegs = indegs
        acyclics = .false.

        ! Process and extend the queue of zero-indegree cells
        iseed = 1
        do while (iseed <= nseeds)
            ci = seed_ijs(1, iseed)
            cj = seed_ijs(2, iseed)
            iseed = iseed + 1

            if (acyclics(ci, cj)) cycle
            acyclics(ci, cj) = .true.

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle

            ! Decrement indegree of downstream cell
            rem_indegs(ni, nj) = rem_indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (rem_indegs(ni, nj) /= 0) cycle

            nseeds = nseeds + 1
            if (nseeds > size(seed_ijs, dim=2)) then
                ! Buffer overflow
                err_code = 3
                deallocate (offset_lookup)
                deallocate (rem_indegs)
                deallocate (seed_ijs)
                return
            end if
            seed_ijs(1, nseeds) = ni
            seed_ijs(2, nseeds) = nj
        end do

        deallocate (offset_lookup)
        deallocate (rem_indegs)
        deallocate (seed_ijs)
    end subroutine find_acyclic_flowdirs

    subroutine compute_max_branch_dist( &
        maxbdists, dirs, valids, x, y, basin_ids, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: basin_ids(nrows, ncols)
            !! Basin ids for checking if two cells belong to the same basin (to skip confluence check)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: maxbdists(nrows, ncols)
            !! Grid of maximum branch distances for each cell, i.e. the maximum distance along flow paths to a confluence point downstream
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A traced flow path contains a cycle
            !!   - 2: Internal workspace allocation failed
            !!   - 3: A traced flow path exceeded its allocated capacity
        ! Local variables
        integer, allocatable :: diffs(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        real :: dists(2)
            !! Array to hold distances from two neighbouring cells to their confluence point
        integer :: nneighbour, neighbour_offsets(4, 2)
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        integer :: maxlen
            !! Maximum path length to search before giving up and assuming no confluence
            !! It should be large enough to allow confluence but prevent infinite loops in case of errors.
        integer :: path1id, path2id
            !! IDs of the first and second paths to check for confluence
            !! When incrementing, each ID is of 'maxlen' apart such that 'path1id + ilen' is unique, which allows for more efficient confluence lookup.
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)
        logical*1, allocatable :: is_max_dist(:, :)
            !! Mask of cells whose maximum branch distance has been finalized
        logical*1 :: same_basin, ci_is_max_dist, ni_is_max_dist
        integer :: alloc_stat
            !! Per-thread allocation status code
        integer :: inner_err_code
            !! Status returned by a confluence trace

        ! Define neighbour offsets
        parameter(neighbour_offsets= &
                  reshape([1, -1, & ! SW
                           0, 1, & ! E
                           1, 1, & ! SE
                           1, 0 & ! S
                           ], [4, 2]))

        ! Create lookup tables for offsets
        err_code = 0
        allocate (diffs(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        diffs = fill_offset_lookup(offsets, codes)

        maxlen = 2*(nrows + ncols)

        allocate (is_max_dist(nrows, ncols), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            deallocate (diffs)
            return
        end if
        is_max_dist = .false.
        maxbdists = 0.0
        !$omp PARALLEL DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, dists, same_basin) &
        !$omp PRIVATE(ci_is_max_dist, ni_is_max_dist) &
        !$omp PRIVATE(path1, path2, path1id, path2id, visited, alloc_stat, inner_err_code)
        allocate (path1(2, maxlen), path2(2, maxlen), &
                  visited(nrows, ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        else
            visited = 0
        end if
        path1id = 1
        path2id = 1 + maxlen
        !$omp DO SCHEDULE(DYNAMIC) &
        !$omp COLLAPSE(2)
        do cj = 1, ncols
            do ci = 1, nrows
                if (alloc_stat /= 0) cycle
                do nneighbour = 1, size(neighbour_offsets, 1)
                    if (.not. valids(ci, cj)) cycle
                    ni = ci + neighbour_offsets(nneighbour, 1)
                    nj = cj + neighbour_offsets(nneighbour, 2)
                    ! Check bounds
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ni, nj)) cycle
                    !$omp ATOMIC READ
                    ci_is_max_dist = is_max_dist(ci, cj)
                    !$omp END ATOMIC
                    !$omp ATOMIC READ
                    ni_is_max_dist = is_max_dist(ni, nj)
                    !$omp END ATOMIC
                    if (ci_is_max_dist .and. ni_is_max_dist) cycle
                    same_basin = logical(basin_ids(ni, nj) == basin_ids(ci, cj), kind=1)
                    call inner_compute_confluence_dist( &
                        dists, ci, cj, ni, nj, dirs, x, y, diffs, &
                        maxpathlen=maxlen, path1=path1, path2=path2, &
                        visited=visited, id1=path1id, id2=path2id, &
                        check_flag=same_basin, &
                        err_code=inner_err_code)
                    if (inner_err_code /= 0) then
                        !$omp critical
                        if (err_code == 0) err_code = inner_err_code
                        !$omp end critical
                        cycle
                    end if
                    !$omp ATOMIC UPDATE
                    maxbdists(ci, cj) = max(maxbdists(ci, cj), dists(1))
                    !$omp END ATOMIC
                    !$omp ATOMIC UPDATE
                    maxbdists(ni, nj) = max(maxbdists(ni, nj), dists(2))
                    !$omp END ATOMIC

                    if (.not. same_basin) then
                        !$omp ATOMIC WRITE
                        is_max_dist(ci, cj) = .true.
                        !$omp END ATOMIC
                        !$omp ATOMIC WRITE
                        is_max_dist(ni, nj) = .true.
                        !$omp END ATOMIC
                    end if

                    if (path1id > 2147483640 - 2*maxlen) then
                        visited = 0
                        path1id = 1
                        path2id = 1 + maxlen
                    end if
                    path1id = path1id + 2*maxlen
                    path2id = path2id + 2*maxlen
                end do
            end do
        end do
        !$omp END DO
        if (allocated(path1)) deallocate (path1)
        if (allocated(path2)) deallocate (path2)
        if (allocated(visited)) deallocate (visited)
        !$omp END PARALLEL
        deallocate (is_max_dist)
        deallocate (diffs)
    end subroutine compute_max_branch_dist

    pure subroutine compute_confluence_dist( &
        dists, &
        s1ij, s2ij, dirs, x, y, &
        offset_lookup, check_flag, err_code)
        !! Traces flow paths from two seed cells downstream to compute their confluence distance.
        implicit none
        ! Arguments
        integer, intent(in) :: s1ij(2), s2ij(2)
            !! (i, j) indices of the two seed cells from which to trace flow paths
        integer*1, intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical*1, intent(in), optional :: check_flag
            !! Whether to check for confluence at each step
        ! Outputs
        real, intent(out) :: dists(2)
            !! Distances from each seed ceel to the confluence cell (or to max path length if no confluence found)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A traced flow path contains a cycle
            !!   - 2: Internal workspace allocation failed
            !!   - 3: A traced flow path exceeded its allocated capacity
        ! Local variables
        logical*1 :: check_flag_
        integer :: maxpathlen
            !! Maximum path length to search before giving up and assuming no confluence
            !! It should be large enough to allow confluence but prevent infinite loops in case of errors.
        integer :: id1, id2
            !! IDs of the first and second paths to check for confluence
            !! When incrementing, each ID is of 'maxpathlen' apart such that 'path1id + ilen' is unique, which allows for more efficient confluence lookup.
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)

        maxpathlen = 4*(size(dirs, 1) + size(dirs, 2))
        id1 = 1
        ! Storing path step offset in visited grid requires id1 and id2 to have disjoint active value ranges.
        ! id1 uses values in [id1, id1 + maxpathlen - 1].
        ! id2 uses values in [id2, id2 + maxpathlen - 1].
        id2 = 1 + maxpathlen
        err_code = 0
        allocate (path1(2, maxpathlen), path2(2, maxpathlen), &
                  visited(size(dirs, 1), size(dirs, 2)), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            dists = 0.0
            return
        end if
        visited = 0

        check_flag_ = (.not. present(check_flag)) .or. check_flag

        call inner_compute_confluence_dist( &
            dists, &
            s1ij(1), s1ij(2), s2ij(1), s2ij(2), dirs, x, y, offset_lookup, &
            maxpathlen, path1, path2, visited, id1, id2, &
            check_flag=check_flag_, err_code=err_code)
        deallocate (path1)
        deallocate (path2)
        deallocate (visited)
    end subroutine compute_confluence_dist

    pure subroutine inner_compute_confluence_dist( &
        dists, s1i, s1j, s2i, s2j, dirs, x, y, &
        offset_lookup, maxpathlen, path1, path2, visited, id1, id2, check_flag, err_code)
        !! Inner routine for computing the confluence distance between two seed cells.
        !!
        !! The 'visited' grid tracks cell visits. It stores the exact path step
        !! index: 'id + ipath - 1'. If 'visited(n1i, n1j)' is in the range
        !! [id2, id2 + npath2 - 1], it means Path 2 has already visited this
        !! cell, and the index at which the confluence occurs in Path 2 is then
        !! retrieved instantly via. This avoids an O(N) linear search over the
        !! path.
        implicit none
        ! Inputs
        integer, intent(in) :: s1i, s1j, s2i, s2j
            !! Indices of the two seed cells from which to trace flow paths
        integer*1, intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, contiguous, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical*1, intent(in), optional :: check_flag
            !! Flag for whether to check for confluence at each step (can be turned off for performance if many confluences are expected)
        integer, intent(in) :: maxpathlen
            !! Maximum path length to search before giving up and assuming no confluence
            !! It should be large enough to allow confluence but prevent infinite loops in case of errors.
        integer, intent(in) :: id1, id2
            !! Unique ids to mark visited cells for each path in the visited grid
        integer, intent(inout) :: path1(2, maxpathlen), path2(2, maxpathlen)
            !! Workspace arrays for paths and visited grid
        integer, contiguous, intent(inout) :: visited(:, :)
            !! Grid to track visited paths by ids
        ! Outputs
        real, intent(out) :: dists(2)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A traced flow path contains a cycle
            !!   - 3: A traced flow path exceeded its allocated capacity
            !! Distances from each seed cell to the confluence cell (or to max path length if no confluence found)
        ! Local variables
        integer :: ipath1, ipath2, npath1, npath2
            !! Indices for iterating through paths and current path lengths
        integer :: iconf1, iconf2
            !! Indices of confluence in paths (or max path length if no confluence found)
        integer :: n1i, n1j, n2i, n2j
            !! Indices of next cell in path for each seed
        integer*1 :: code1, code2
            !! Flow direction codes for current cells in paths
        logical*1 :: is_active1, is_active2, local_check_flag
            !! Flags for whether each path is still active (has not reached max length or invalid cell) and local copy of check_flag for performance

        !! Initialisation and checks
        local_check_flag = (.not. present(check_flag)) .or. check_flag
        iconf1 = maxpathlen
        iconf2 = maxpathlen

        dists = 0.0
        err_code = 0
        is_active1 = .true.
        is_active2 = .true.

        ! Return zero if same cell
        if ((s1i == s2i) .and. (s1j == s2j)) then
            dists = 0.0
            return
        end if

        !! Main algorithm
        npath1 = 1
        path1(1, npath1) = s1i
        path1(2, npath1) = s1j
        visited(s1i, s1j) = id1
        npath2 = 1
        path2(1, npath2) = s2i
        path2(2, npath2) = s2j
        visited(s2i, s2j) = id2

        tracer_loop: do while (is_active1 .or. is_active2)
            path1_prc: block
                if (.not. is_active1) exit path1_prc
                ! Make sure code is valid
                code1 = dirs(path1(1, npath1), path1(2, npath1))
                if (code1 < lbound(offset_lookup, 1) .or. code1 > ubound(offset_lookup, 1)) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                else if (offset_lookup(code1, 1) == 0 .and. offset_lookup(code1, 2) == 0) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                end if

                ! Compute next step
                n1i = path1(1, npath1) + offset_lookup(code1, 1)
                n1j = path1(2, npath1) + offset_lookup(code1, 2)
                if (n1i < 1 .or. n1i > size(dirs, 1) .or. n1j < 1 .or. n1j > size(dirs, 2)) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                else if (npath1 >= maxpathlen) then
                    err_code = 3
                    iconf1 = npath1
                    return
                end if
                npath1 = npath1 + 1
                path1(1, npath1) = n1i
                path1(2, npath1) = n1j
                ! Check for self-intersection (value lies within Path 1's active range of IDs for the current run)
                if (visited(n1i, n1j) >= id1 .and. visited(n1i, n1j) < id1 + npath1 - 1) then
                    err_code = 1
                    iconf1 = npath1
                    return
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path1_prc
                ! If the cell wasn't visited by Path 2 in the current run (value not in Path 2's ID range),
                ! mark it with Path 1's base ID + step offset, and continue tracing.
                if (visited(n1i, n1j) < id2 .or. visited(n1i, n1j) >= id2 + npath2) then
                    visited(n1i, n1j) = id1 + npath1 - 1
                    exit path1_prc
                end if
                ! Confluence found: retrieve the exact matching index in Path 2 in O(1) time
                iconf1 = npath1
                iconf2 = visited(n1i, n1j) - id2 + 1
                exit tracer_loop
            end block path1_prc

            path2_prc: block
                if (.not. is_active2) exit path2_prc
                ! Make sure code is valid
                code2 = dirs(path2(1, npath2), path2(2, npath2))
                if (code2 < lbound(offset_lookup, 1) .or. code2 > ubound(offset_lookup, 1)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (offset_lookup(code2, 1) == 0 .and. offset_lookup(code2, 2) == 0) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                end if
                n2i = path2(1, npath2) + offset_lookup(code2, 1)
                n2j = path2(2, npath2) + offset_lookup(code2, 2)
                if (n2i < 1 .or. n2i > size(dirs, 1) .or. n2j < 1 .or. n2j > size(dirs, 2)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (npath2 >= maxpathlen) then
                    err_code = 3
                    iconf2 = npath2
                    return
                end if
                npath2 = npath2 + 1
                path2(1, npath2) = n2i
                path2(2, npath2) = n2j
                ! Check for self-intersection (value lies within Path 2's active range of IDs for the current run)
                if (visited(n2i, n2j) >= id2 .and. visited(n2i, n2j) < id2 + npath2 - 1) then
                    err_code = 1
                    iconf2 = npath2
                    return
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path2_prc
                ! If the cell wasn't visited by Path 1 in the current run (value not in Path 1's ID range),
                ! mark it with Path 2's base ID + step offset, and continue tracing.
                if (visited(n2i, n2j) < id1 .or. visited(n2i, n2j) >= id1 + npath1) then
                    visited(n2i, n2j) = id2 + npath2 - 1
                    exit path2_prc
                end if
                ! Confluence found: retrieve the exact matching index in Path 1 in O(1) time
                iconf1 = visited(n2i, n2j) - id1 + 1
                iconf2 = npath2
                exit tracer_loop
            end block path2_prc
        end do tracer_loop

        ! Compute distances to confluence
        do ipath1 = 1, min(iconf1, npath1) - 1
            dists(1) = dists(1) + l2dist_xy( &
                       x(path1(1, ipath1 + 1), path1(2, ipath1 + 1)), &
                       y(path1(1, ipath1 + 1), path1(2, ipath1 + 1)), &
                       x(path1(1, ipath1), path1(2, ipath1)), &
                       y(path1(1, ipath1), path1(2, ipath1)))
        end do
        do ipath2 = 1, min(iconf2, npath2) - 1
            dists(2) = dists(2) + l2dist_xy( &
                       x(path2(1, ipath2 + 1), path2(2, ipath2 + 1)), &
                       y(path2(1, ipath2 + 1), path2(2, ipath2 + 1)), &
                       x(path2(1, ipath2), path2(2, ipath2)), &
                       y(path2(1, ipath2), path2(2, ipath2)))
        end do
    end subroutine inner_compute_confluence_dist

    ! subroutine compute_spill_flow( &
    !     z, valids, flowdirs, nrows, ncols, &
    !     offsets, codes, noffsets)
    !     implicit none
    !     ! Inputs
    !     integer, intent(in) :: nrows, ncols
    !     real, dimension(nrows, ncols), intent(in) :: z
    !     logical, dimension(nrows, ncols), intent(in) :: valids
    !     integer, intent(in) :: noffsets
    !     integer, dimension(noffsets, 2), intent(in) :: offsets
    !     integer*1, dimension(noffsets), intent(in) :: codes
    !     ! Outputs
    !     integer*1, dimension(nrows, ncols), intent(out) :: flowdirs

    !     logical, allocatable :: processed(:, :)
    !     integer*1, allocatable :: indegs(:, :)
    !     integer, allocatable :: dists(:, :)
    !     integer*1, dimension(noffsets) :: opp_codes
    !     integer*1 :: noflow_code = 0

    !     integer :: sij(2) ! Seed indices

    !     noflow_code = find_noflow_code(offsets, codes)
    !     opp_codes = find_opposite_codes(offsets, codes)

    !     allocate (processed(nrows, ncols))
    !     call compute_flowdir_simple( &
    !         z, valids, flowdirs, processed, nrows, ncols, &
    !         offsets, codes, noffsets)

    !     processed = .false.
    !     ! Fill invalid cells as processed
    !     processed = merge(.true., processed,.not. valids)
    !     ! Fill boundary cells as processed
    !     processed(1, :) = .true.
    !     processed(nrows, :) = .true.
    !     processed(:, 1) = .true.
    !     processed(:, ncols) = .true.
    !     call flood_upstream( &
    !         processed, flowdirs, processed, valids, nrows, ncols, &
    !         offsets, codes, noffsets)

    !     allocate (dists(nrows, ncols))
    !     call compute_dist2source_l1( &
    !         flowdirs, valids, indegs, dists, nrows, ncols, &
    !         offsets, codes, noffsets)

    !     if (count(processed) == nrows*ncols) then
    !         ! All cells processed
    !         deallocate (processed)
    !         deallocate (dists)
    !         return
    !     end if

    !     ! Find seed: min elevation among unprocessed cells
    !     sij = minloc(z, mask=.not. processed)
    !     ! Find lowest border cell of the basin containing the seed

    ! end subroutine compute_spill_flow
end module flowdir_raster
