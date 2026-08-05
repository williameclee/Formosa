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
!     - Explicitly handled Python uint8 -> signed 8-bit Fortran conversion/interpretation in 'fill_offset_lookup'
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Added allocation error monitoring and moved error handling to Python
!     - Used function 'mask2id' as the linear-index version of 'mask2ij'
!   2026-08-05, En-Chi Lee (williameclee@gmail.com)
!     - Overhauled algorithm for 'compute_max_branch_dist'
!     - Switched to 'iso_c_binding'
!!!

module flowdir_raster
    use iso_c_binding, only: c_int8_t, c_int16_t
    use utils, only: fill_offset_lookup, find_noflow_code, id2ij_checked, &
        ij2id_checked, mask2id, mask2ij
    use distances, only: l1dist_xy, l2dist_xy
    implicit none(type, external)
    private :: resolve_flow_tree_links, build_flow_tree_topology
    private :: propagate_flow_tree_metadata, build_flow_tree_metadata
    private :: find_tree_confluence
contains
    subroutine compute_flowdir_simple( &
        z, valids, dirs, is_flat, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a given elevation grid, using
        !! the provided flow direction codes and offsets.
        !! Also identifies flat cells where no flow direction can be
        !! assigned.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer(c_int8_t), intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(out) :: is_flat(nrows, ncols)
            !! Mask indicating which cells are part of flats (i.e. direction is no-flow)
        ! Local variables
        integer(c_int8_t) :: noflow_code
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
        implicit none(type, external)
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
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer(c_int8_t), intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        ! Local variables
        integer(c_int8_t) :: noflow_code
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical(kind=1), intent(out) :: is_low_edge(nrows, ncols), is_high_edge(nrows, ncols)
            !! Whether each cell is a 'low edge' or a 'high edge of a flat.
        ! Local variables
        integer(c_int8_t) :: noflow_code
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: seeds(nrows, ncols)
            !! Seed mask indicating starting points for labelling flat regions
        logical(kind=1), intent(in) :: valids(nrows, ncols)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
        logical(kind=1), intent(in) :: high_edges(nrows, ncols)
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
        logical(kind=1), allocatable :: queued(:, :)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer, intent(in) :: flats(nrows, ncols)
            !! Label grid indicating individual flat regions (or 0 for non-flat cells)
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
        logical(kind=1), allocatable :: queued(:, :)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer(c_int8_t), intent(out) :: indegs(nrows, ncols)
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
                        indegs(ci, cj) = indegs(ci, cj) + int(1, kind=c_int8_t)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        real, intent(in) :: areas(nrows, ncols)
            !! Area of each cell, used as the initial accumulation value for each cell
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell.
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
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
        logical(kind=1), allocatable :: flood_seeds(:, :)
            !! Mask to identify initial seed cells for the flooding algorithm (valid cells with zero in-degrees)

        ! Guard nrows*ncols before using it as a default-integer allocation
        ! extent or in the column-major linear-index expressions below.
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        ! Convert arbitrary external direction codes into an O(1) lookup table.
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
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=c_int8_t)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
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
        logical(kind=1), allocatable :: tofill_seeds(:, :)
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
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=c_int8_t)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
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
        logical(kind=1), allocatable :: seeds(:, :)
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
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=c_int8_t)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction grid encoded using codes.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate flow-edge distances.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells participating in the flow forest; false for no-data.
        integer, intent(in) :: noffsets
            !! Number of supported flow-direction codes.
        integer, intent(in) :: offsets(noffsets, 2)
            !! Row/column displacement corresponding to each direction code.
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! External direction codes corresponding to offsets.
        ! Outputs
        real, intent(out) :: dists(nrows, ncols)
            !! Grid of distances to the downstream sink
        integer, intent(out) :: err_code
            !! Code indicating the status of the result:
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Sink-distance queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to noflow direction, used to identify sink cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream cells
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid cells with noflow direction)
        integer, allocatable :: seed_ids(:)
            !! Buffer for storing linear IDs of seed cells to be processed in the algorithm
        integer, allocatable :: tofill_ids(:)
            !! Buffer for storing linear cell IDs in the breadth-first search from sink cells
        integer :: cell_id
        logical(kind=1) :: id_is_valid
        integer :: max_queue_size
            !! Maximum number of linear cell IDs in the seed and traversal queues
        integer :: alloc_stat
            !! Per-thread allocation status code
        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        dists = -1

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ids(max_queue_size), stat=err_code)
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
        call mask2id(seeds, seed_ids, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ui, uj, iofs) &
        !$omp PRIVATE(ifill, nfills, tofill_ids, alloc_stat) &
        !$omp PRIVATE(cell_id, id_is_valid)
        allocate (tofill_ids(max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            call id2ij_checked(seed_ids(iseed), nrows, ncols, si, sj, id_is_valid)
            if (.not. id_is_valid) then
                !$omp atomic write
                err_code = 3
                cycle
            end if

            ! Loop through buffer
            nfills = 1
            ifill = 1
            dists(si, sj) = 0.0
            tofill_ids(1) = seed_ids(iseed)

            do while (ifill <= nfills)
                call id2ij_checked(tofill_ids(ifill), nrows, ncols, ci, cj, id_is_valid)
                ifill = ifill + 1
                if (.not. id_is_valid) then
                    !$omp atomic write
                    err_code = 3
                    exit
                end if

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
                    cell_id = ij2id_checked(ui, uj, nrows, ncols)
                    if (cell_id == 0) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ids(nfills) = cell_id
                    ! Compute distance
                    dists(ui, uj) = dists(ci, cj) &
                                    + l2dist_xy(x(ui, uj), y(ui, uj), x(ci, cj), y(ci, cj))
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ids)) deallocate (tofill_ids)
        !$omp END PARALLEL
        deallocate (seed_ids)
    end subroutine compute_dist2sink

    subroutine compute_flow_strahler_order( &
        dirs, valids, indegs, orders, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that flow into each cell
            !! This will be modified in-place during the algorithm to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        integer(c_int16_t), intent(out) :: orders(nrows, ncols)
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
        integer(c_int16_t) :: max_uorder
            !! Maximum Strahler stream order value of a cell's upstream neighbours
        logical(kind=1) :: increase_order
            !! Whether the current cell's order should be increased
        logical(kind=1), allocatable :: seeds(:, :)
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
        orders = merge(int(1, kind=c_int16_t), int(0, kind=c_int16_t), seeds)
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
                    orders(ci, cj) = max_uorder + int(1, kind=c_int16_t)
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
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=c_int8_t)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
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
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to noflow direction, used to identify seed cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream indices
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid cells with noflow direction)
        integer, allocatable :: seed_ids(:), tofill_ids(:)
            !! Buffers for storing linear IDs of seed cells and queued cells
        integer :: cell_id
        logical(kind=1) :: id_is_valid
        integer :: max_queue_size
            !! Maximum number of linear cell IDs in the seed and traversal queues
        integer :: alloc_stat
            !! Per-thread allocation status code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        labels = 0

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ids(max_queue_size), stat=err_code)
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
        call mask2id(seeds, seed_ids, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ui, uj, iofs) &
        !$omp PRIVATE(ifill, nfills, tofill_ids, alloc_stat) &
        !$omp PRIVATE(cell_id, id_is_valid)
        allocate (tofill_ids(max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            call id2ij_checked(seed_ids(iseed), nrows, ncols, si, sj, id_is_valid)
            if (.not. id_is_valid) then
                !$omp atomic write
                err_code = 3
                cycle
            end if

            ! Loop through buffer
            nfills = 1
            ifill = 1
            labels(si, sj) = iseed
            tofill_ids(1) = seed_ids(iseed)

            do while (ifill <= nfills)
                call id2ij_checked(tofill_ids(ifill), nrows, ncols, ci, cj, id_is_valid)
                ifill = ifill + 1
                if (.not. id_is_valid) then
                    !$omp atomic write
                    err_code = 3
                    exit
                end if

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
                    cell_id = ij2id_checked(ui, uj, nrows, ncols)
                    if (cell_id == 0) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ids(nfills) = cell_id
                    ! Compute distance
                    labels(ui, uj) = labels(ci, cj)
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ids)) deallocate (tofill_ids)
        !$omp END PARALLEL
        deallocate (seed_ids)
    end subroutine label_watersheds

    subroutine flood_upstream( &
        flooded, dirs, seeds, valids, nrows, ncols, offsets, codes, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols), seeds(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data) and seed mask (true for seed cells, false for non-seed cells)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical(kind=1), intent(out) :: flooded(nrows, ncols)
            !! Mask indicating which cells are flooded (true for flooded cells, false for non-flooded cells)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Upstream-flooding queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to noflow direction, used to identify seed cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer, and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream indices
        integer, allocatable :: seed_ids(:), tofill_ids(:)
            !! Buffers for storing linear IDs of seed cells and queued cells
        integer :: cell_id
        logical(kind=1) :: id_is_valid
        integer :: max_queue_size
            !! Maximum number of linear cell IDs in the seed and traversal queues
        integer :: alloc_stat
            !! Per-thread allocation status code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        err_code = 0
        flooded = .false.

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ids(max_queue_size), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        call mask2id(seeds, seed_ids, max_queue_size, nseeds, err_code)
        if (err_code /= 0) return

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ui, uj, iofs) &
        !$omp PRIVATE(ifill, nfills, tofill_ids, alloc_stat) &
        !$omp PRIVATE(cell_id, id_is_valid)
        allocate (tofill_ids(max_queue_size), stat=alloc_stat)
        if (alloc_stat /= 0) then
            !$omp atomic write
            err_code = 2
        end if
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            if (alloc_stat /= 0) cycle
            call id2ij_checked(seed_ids(iseed), nrows, ncols, si, sj, id_is_valid)
            if (.not. id_is_valid) then
                !$omp atomic write
                err_code = 3
                cycle
            end if

            ! Check if is valid
            if (.not. valids(si, sj)) cycle

            ! Loop through buffer
            nfills = 1
            ifill = 1
            flooded(si, sj) = .true.
            tofill_ids(1) = seed_ids(iseed)

            do while (ifill <= nfills)
                call id2ij_checked(tofill_ids(ifill), nrows, ncols, ci, cj, id_is_valid)
                ifill = ifill + 1
                if (.not. id_is_valid) then
                    !$omp atomic write
                    err_code = 3
                    exit
                end if

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
                    cell_id = ij2id_checked(ui, uj, nrows, ncols)
                    if (cell_id == 0) then
                        !$omp atomic write
                        err_code = 3
                        exit
                    end if
                    tofill_ids(nfills) = cell_id
                    ! Compute distance
                    flooded(ui, uj) = .true.
                end do
            end do
        end do
        !$omp END DO
        if (allocated(tofill_ids)) deallocate (tofill_ids)
        !$omp END PARALLEL
        deallocate (seed_ids)
    end subroutine flood_upstream

    subroutine find_acyclic_flowdirs( &
        dirs, indegs, valids, nrows, ncols, offsets, codes, noffsets, acyclics, err_code)
        !! Identifies valid cells that are not part of a directed flow cycle.
        !! Uses Kahn's algorithm to traverse cells from zero-indegree seeds,
        !! successively removing their outgoing edges. Valid cells not reached
        !! by this traversal belong to a directed cycle and remain false in
        !! 'acyclics'.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        integer(c_int8_t), intent(in) :: indegs(nrows, ncols)
            !! Indegree grid for the valid flow field
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        logical(kind=1), intent(out) :: acyclics(nrows, ncols)
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
        integer(c_int8_t), allocatable :: rem_indegs(:, :)
            !! Remaining indegrees after removing edges from processed cells
        logical(kind=1), allocatable :: seeds(:, :)
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
            rem_indegs(ni, nj) = rem_indegs(ni, nj) - int(1, kind=c_int8_t)
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

    subroutine resolve_flow_tree_links( &
        dirs, valids, offset_lookup, nrows, ncols, ds_ids, indegs)
        !! Resolve each valid cell's immediate downstream ID and simultaneously
        !! count the upstream children of every destination cell.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction code for every raster cell.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells belonging to the flow tree; false for no-data.
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Row/column offset indexed by the unsigned direction code.
        integer, intent(out) :: ds_ids(nrows*ncols)
            !! Immediate downstream cell ID, or zero at a sink.
        integer(c_int8_t), intent(out) :: indegs(nrows*ncols)
            !! Number of valid upstream children targeting each cell.
        integer :: code
            !! Unsigned integer representation of the current direction code.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream destination.
        integer :: ci, cj, ni, nj
            !! Row/column coordinates of the current and downstream cells.

        ds_ids = 0
        indegs = 0

        ! Resolve one downstream edge per valid source cell. Coordinates are
        ! checked before linear encoding so an invalid pair cannot wrap into a
        ! different, apparently legitimate cell ID.
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, code, cid, dsid) &
        !$omp COLLAPSE(2) SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                cid = ij2id_checked(ci, cj, nrows, ncols)
                code = iand(int(dirs(ci, cj)), 255)
                if (offset_lookup(code, 1) == -99 .and. offset_lookup(code, 2) == -99) cycle
                ni = ci + offset_lookup(code, 1)
                nj = cj + offset_lookup(code, 2)
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                if (.not. valids(ni, nj)) cycle
                dsid = ij2id_checked(ni, nj, nrows, ncols)
                if (dsid == 0 .or. dsid == cid) cycle
                ds_ids(cid) = dsid
                !$omp ATOMIC UPDATE
                indegs(dsid) = indegs(dsid) + int(1, kind=c_int8_t)
                !$omp END ATOMIC
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine resolve_flow_tree_links

    subroutine build_flow_tree_topology( &
        valids, ds_ids, indegs, nrows, ncols, &
        topo_order, topo_cnt, lvl_ends, nlvls, err_code)
        !! Build source-to-sink Kahn frontiers and reject directed cycles.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells belonging to the flow tree; false for no-data.
        integer, intent(in) :: ds_ids(nrows*ncols)
            !! Immediate downstream ID for every cell; zero at sinks.
        integer(c_int8_t), intent(inout) :: indegs(nrows*ncols)
            !! Remaining unprocessed upstream-child count for Kahn traversal.
        integer, intent(out) :: topo_order(nrows*ncols)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        integer, intent(out) :: topo_cnt
            !! Number of valid entries written to topo_order.
        integer, allocatable, intent(inout) :: lvl_ends(:)
            !! Inclusive topo_order end position of every Kahn frontier.
        integer, intent(out) :: nlvls
            !! Number of dependency frontiers recorded in lvl_ends.
        integer, intent(out) :: err_code
            !! Zero on success, one for a cycle, or two for allocation failure.
        ! Local variables
        integer, allocatable :: grown_lvl_ends(:)
            !! Temporary buffer used when geometrically growing lvl_ends.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream destination.
        integer :: ci, cj
            !! Row/column coordinates used while identifying valid sources.
        integer :: sorder
            !! Current read position of topo_order.
        integer :: lvl_start, lvl_end, next_lvl_end
            !! Inclusive bounds of the active Kahn frontier and its appended end.
        integer :: nvalid
            !! Number of valid cells expected in a complete topological order.
        integer :: new_lvl_capacity, alloc_stat
            !! Requested level-buffer capacity and allocation status.

        err_code = 0
        topo_cnt = 0
        lvl_end = 0
        nvalid = 0

        ! Count number of valid cells and push 0-indegree cells into 'topo_order'
        do cid = 1, nrows*ncols
            ci = mod(cid - 1, nrows) + 1
            cj = (cid - 1)/nrows + 1
            if (.not. valids(ci, cj)) cycle
            nvalid = nvalid + 1
            if (indegs(cid) /= 0) cycle
            lvl_end = lvl_end + 1
            topo_order(lvl_end) = cid
        end do

        nlvls = 0
        lvl_start = 1
        lvl_end = lvl_end
        ! Go through cells level by level
        do while (lvl_start <= lvl_end)
            nlvls = nlvls + 1

            ! Reallocate lvl_ends if needed
            if (nlvls > size(lvl_ends)) then
                new_lvl_capacity = min(size(lvl_ends)*2, nrows*ncols)
                allocate (grown_lvl_ends(new_lvl_capacity), stat=alloc_stat)
                if (alloc_stat /= 0) then
                    err_code = 2
                    return
                end if
                grown_lvl_ends(1:nlvls - 1) = lvl_ends
                call move_alloc(grown_lvl_ends, lvl_ends)
            end if

            lvl_ends(nlvls) = lvl_end
            next_lvl_end = lvl_end

            ! Sweep through all the downstream cells of this frontier
            do sorder = lvl_start, lvl_end
                cid = topo_order(sorder)
                dsid = ds_ids(cid)
                if (dsid == 0) cycle
                indegs(dsid) = indegs(dsid) - int(1, kind=c_int8_t)
                if (indegs(dsid) /= 0) cycle
                next_lvl_end = next_lvl_end + 1
                topo_order(next_lvl_end) = dsid
            end do
            lvl_start = lvl_end + 1
            lvl_end = next_lvl_end
        end do
        topo_cnt = lvl_end
        if (topo_cnt /= nvalid) err_code = 1
    end subroutine build_flow_tree_topology

    subroutine propagate_flow_tree_metadata( &
        ds_ids, x, y, topo_order, lvl_ends, nlvls, nrows, ncols, &
        depths, sink_ids, sink_dists)
        !! Propagates depth, sink identity, and metric distance from sinks towards
        !! sources in reverse dependency-frontier order.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols, nlvls
            !! Raster dimensions and number of dependency frontiers.
        integer, intent(in) :: ds_ids(nrows*ncols)
            !! Immediate downstream ID for every cell; zero at sinks.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate metric edge lengths.
        integer, intent(in) :: topo_order(nrows*ncols), lvl_ends(:)
            !! Source-to-sink cell order and inclusive frontier boundaries.
        integer, intent(out) :: depths(nrows*ncols)
            !! Number of downstream edges from each cell to its sink.
        integer, intent(out) :: sink_ids(nrows*ncols)
            !! Linear ID of the sink reached by each valid cell.
        real, intent(out) :: sink_dists(nrows*ncols)
            !! Cumulative metric distance from each valid cell to its sink.
        integer, parameter :: min_parallel_lvl = 32768
            !! Minimum frontier width worth entering an OpenMP parallel region.
        integer :: ilvl, lvl_start, lvl_end, sorder
            !! Current frontier, its inclusive bounds, and traversal cursor.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream parent.
        integer :: ci, cj, dsi, dsj
            !! Row/column coordinates of the current cell and its downstream parent.

        depths = 0
        sink_ids = 0
        sink_dists = 0.0
        do ilvl = nlvls, 1, -1
            lvl_end = lvl_ends(ilvl)
            if (ilvl == 1) then
                lvl_start = 1
            else
                lvl_start = lvl_ends(ilvl - 1) + 1
            end if

            ! Check if worth parallelising
            if (lvl_end - lvl_start + 1 >= min_parallel_lvl) then
                !$omp PARALLEL DO DEFAULT(SHARED) &
                !$omp PRIVATE(sorder, cid, dsid, ci, cj, dsi, dsj) SCHEDULE(STATIC)
                do sorder = lvl_start, lvl_end
                    ! Calculate distance to sink based on its downstream's distance to sink
                    cid = topo_order(sorder)
                    dsid = ds_ids(cid)
                    if (dsid == 0) then
                        sink_ids(cid) = cid
                        cycle
                    end if
                    depths(cid) = depths(dsid) + 1
                    sink_ids(cid) = sink_ids(dsid)
                    ci = mod(cid - 1, nrows) + 1
                    cj = (cid - 1)/nrows + 1
                    dsi = mod(dsid - 1, nrows) + 1
                    dsj = (dsid - 1)/nrows + 1
                    sink_dists(cid) = sink_dists(dsid) &
                                      + l2dist_xy(x(ci, cj), y(ci, cj), x(dsi, dsj), y(dsi, dsj))
                end do
                !$omp END PARALLEL DO
            else
                ! Same loop as above
                do sorder = lvl_start, lvl_end
                    cid = topo_order(sorder)
                    dsid = ds_ids(cid)
                    if (dsid == 0) then
                        sink_ids(cid) = cid
                        cycle
                    end if
                    depths(cid) = depths(dsid) + 1
                    sink_ids(cid) = sink_ids(dsid)
                    ci = mod(cid - 1, nrows) + 1
                    cj = (cid - 1)/nrows + 1
                    dsi = mod(dsid - 1, nrows) + 1
                    dsj = (dsid - 1)/nrows + 1
                    sink_dists(cid) = sink_dists(dsid) &
                                      + l2dist_xy(x(ci, cj), y(ci, cj), x(dsi, dsj), y(dsi, dsj))
                end do
            end if
        end do
    end subroutine propagate_flow_tree_metadata

    subroutine build_flow_tree_metadata( &
        dirs, valids, x, y, offset_lookup, nrows, ncols, &
        ds_ids, depths, sink_ids, sink_dists, topo_order, topo_cnt, err_code)
        !! Coordinate construction of reusable metadata for the downstream tree.
        !!
        !! The work is deliberately split into three independently testable
        !! phases:
        !!
        !!   resolve_flow_tree_links       -- downstream IDs and indegrees;
        !!   build_flow_tree_topology      -- Kahn frontiers and cycle detection;
        !!   propagate_flow_tree_metadata  -- depths, sinks, and distances.
        !!
        !! topo_order is returned because compute_max_branch_dist reuses it to
        !! construct lowest common ancestor (LCA) jump pointers before releasing 
        !! the full-grid workspace.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction code for every raster cell.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells belonging to the flow tree; false for no-data.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate metric edge lengths.
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Row/column offset indexed by the unsigned direction code.
        integer, intent(out) :: ds_ids(nrows*ncols)
            !! Immediate downstream parent ID, or zero when the cell is a sink.
        integer, intent(out) :: depths(nrows*ncols)
            !! Number of downstream edges from each cell to its sink.
        integer, intent(out) :: sink_ids(nrows*ncols)
            !! Linear ID of the sink reached by each valid cell.
        real, intent(out) :: sink_dists(nrows*ncols)
            !! Cumulative metric distance from each valid cell to its sink.
        integer, allocatable, intent(out) :: topo_order(:)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        integer, intent(out) :: topo_cnt
            !! Number of valid entries written to topo_order.
        integer, intent(out) :: err_code
            !! Zero on success, one for a cycle, or two for allocation failure.
        integer(c_int8_t), allocatable :: indegs(:)
            !! Mutable upstream-child counts consumed by Kahn traversal.
        integer, allocatable :: lvl_ends(:)
            !! Inclusive topo_order end position for each dependency frontier.
        integer :: nlvls
            !! Number of dependency frontiers recorded in lvl_ends.
        integer :: alloc_stat
            !! Status returned by workspace allocation statements.

        err_code = 0
        topo_cnt = 0
        allocate (indegs(nrows*ncols), topo_order(nrows*ncols), &
                  lvl_ends(max(1, min(nrows*ncols, 1024))), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            if (allocated(indegs)) deallocate (indegs)
            if (allocated(topo_order)) deallocate (topo_order)
            if (allocated(lvl_ends)) deallocate (lvl_ends)
            return
        end if

        call resolve_flow_tree_links( &
            dirs, valids, offset_lookup, nrows, ncols, ds_ids, indegs)

        call build_flow_tree_topology( &
            valids, ds_ids, indegs, nrows, ncols, &
            topo_order, topo_cnt, lvl_ends, nlvls, err_code)
        if (err_code /= 0) then
            deallocate (indegs, topo_order, lvl_ends)
            return
        end if

        call propagate_flow_tree_metadata( &
            ds_ids, x, y, topo_order, lvl_ends, nlvls, nrows, ncols, &
            depths, sink_ids, sink_dists)

        deallocate (indegs, lvl_ends)
    end subroutine build_flow_tree_metadata

    pure function find_tree_confluence(cid1, cid2, ds_ids, depths, jump_ids) result(confluence_id)
        !! Returns the first common downstream cell using depth-block jumps.
        !!
        !! depth is measured from a cell downstream to its sink. jump_ids(v)
        !! identifies the anchor at the top of v's fixed-size depth block. The
        !! first loop skips whole blocks from whichever node has the deeper
        !! anchor. Once both nodes share an anchor, the second loop follows
        !! individual parent edges until they meet.
        !!
        !! For maximum tree depth D and block size B this changes a worst-case
        !! O(D) parent walk into approximately O(D/B + B), while needing only
        !! one jump integer per cell rather than the O(N log D) storage required
        !! by binary lifting.
        !!
        !! Precondition: cell 1 and cell 2 belong to the same sink tree. The caller
        !! establishes this without a per-query sink comparison: every pair of
        !! adjacent cells in different trees has both endpoints marked as basin
        !! boundary cells and is skipped before calling this function.
        implicit none(type, external)
        integer, intent(in) :: cid1, cid2
            !! Linear IDs of the two cells to process.
        integer, intent(in) :: ds_ids(:)
            !! Immediate downstream parent ID for every cell; zero at sinks.
        integer, intent(in) :: depths(:)
            !! Number of downstream edges between each cell and its sink.
        integer, intent(in) :: jump_ids(:)
            !! Depth-block anchor ID used to skip groups of parent edges.
        integer :: confluence_id
            !! Linear ID of the first common downstream cell; zero on invalid input.
        integer :: pid1, pid2
            !! Mutable downstream cursors used while aligning and joining paths.

        confluence_id = 0
        if (cid1 < 1 .or. cid1 > size(ds_ids)) return
        if (cid2 < 1 .or. cid2 > size(ds_ids)) return
        pid1 = cid1
        pid2 = cid2
        do while (jump_ids(pid1) /= jump_ids(pid2))
            if (depths(jump_ids(pid1)) > depths(jump_ids(pid2))) then
                pid1 = ds_ids(jump_ids(pid1))
            else
                pid2 = ds_ids(jump_ids(pid2))
            end if
        end do
        do while (pid1 /= pid2)
            ! Walk the branch farther from the sink
            if (depths(pid1) > depths(pid2)) then
                pid1 = ds_ids(pid1)
            else if (depths(pid2) > depths(pid1)) then
                pid2 = ds_ids(pid2)
            else
                pid1 = ds_ids(pid1)
                pid2 = ds_ids(pid2)
            end if
        end do
        confluence_id = pid1
    end function find_tree_confluence

    subroutine compute_max_branch_dist( &
        maxbdists, dirs, valids, x, y, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes, for every valid cell, the largest distance from that cell to
        !! its first downstream confluence with any of its eight neighbours.
        !! If a neighbour belongs to another sink tree, the two paths never
        !! converge and the cell's complete distance to its sink is considered.
        !!
        !! The implementation has four phases:
        !!
        !!  1. Build the downstream forest and cumulative sink metadata.
        !!  2. Mark cells touching a different sink tree. Their answer is known
        !!     immediately to be their complete sink distance.
        !!  3. Reuse the no-longer-needed sink-ID array for depth-block jump
        !!     pointers used by lowest-common-ancestor searches.
        !!  4. Examine each undirected neighbour edge once and atomically update
        !!     the maximum for its two endpoints.
        !!
        !! The tree representation avoids tracing two complete flow paths for
        !! every neighbour pair. It also uses shared O(N) metadata rather than a
        !! full-grid visited/path workspace for every OpenMP thread.
        implicit none(type, external)
        ! Inputs
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: maxbdists(nrows, ncols)
            !! Maximum downstream distance from each cell to its first
            !! confluence with any neighbouring cell.
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: The valid flow field contains a cycle
            !!   - 2: Internal workspace allocation failed
            !!   - 3: A traversal queue exceeded its allocated capacity
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Row/column displacement indexed directly by unsigned code.
        integer, allocatable :: ds_ids(:)
            !! Immediate downstream parent ID for every raster cell.
        integer, allocatable :: depths(:)
            !! Number of downstream edges from each cell to its sink.
        integer, allocatable :: sink_ids(:)
            !! Initially the sink root IDs; later reused as LCA jump pointers.
        integer, allocatable :: topo_order(:)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        real, allocatable :: sink_dists(:)
            !! Cumulative metric distance from each cell to its sink.
        logical(kind=1), allocatable :: is_boundary(:)
            !! True when a cell touches a valid cell belonging to another sink.
        real :: dist1, dist2
            !! Branch distances from the two endpoints of the current grid edge.
        integer :: nneighbour
            !! Index of the neighbour orientation currently being evaluated.
        integer :: neighbour_offsets(4, 2)
            !! Half of the eight-neighbour stencil. These four orientations enumerate every undirected adjacency exactly once.
        integer :: boundary_offsets(8, 2)
            !! Complete eight-neighbour stencil used for boundary classification.
        ! A smaller block shortens the final parent walk but requires more block
        ! jumps. Sixteen was the fastest measured value on the representative
        ! elevation raster and adds no storage of its own.
        integer, parameter :: jump_block_size = 16
            !! Number of depth levels represented by one LCA jump block.
        integer :: ci, cj, ni, nj
            !! Row/column coordinates of the current and neighbouring cells.
        integer :: cid, nid
            !! Linear IDs of the current and neighbouring cells.
        integer :: conf_id
            !! Linear ID of the first common downstream cell for the current pair.
        integer :: alloc_stat
            !! Status returned by workspace allocation statements.
        integer :: topo_cnt
            !! Number of valid entries in topo_order.
        integer :: sorder
            !! Reverse topological cursor used while constructing jump pointers.
        logical :: on_border
            !! True when the current cell needs explicit neighbour bounds checks.

        ! Fortran reshape fills the first dimension first, producing the rows
        ! (1,1), (-1,1), (0,1), and (1,0). Together these cover every undirected
        ! eight-neighbour edge once. The opposite directions would duplicate
        ! both the confluence work and atomic output updates.
        parameter(neighbour_offsets= &
                  reshape([1, -1, &
                           0, 1, &
                           1, 1, &
                           1, 0 &
                           ], [4, 2]))
        ! Boundary classification assigns a property to the current cell only,
        ! so it requires the complete eight-neighbour stencil.
        parameter(boundary_offsets= &
                  reshape([-1, 0, 1, -1, 1, -1, 0, 1, &
                           -1, -1, -1, 0, 0, 1, 1, 1], [8, 2]))

        ! Create lookup tables for offsets
        err_code = 0
        if (nrows < 1 .or. ncols < 1) then
            err_code = 3
            maxbdists = 0.0
            return
        else if (ncols > huge(nrows)/nrows) then
            err_code = 3
            maxbdists = 0.0
            return
        end if
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        allocate (ds_ids(nrows*ncols), depths(nrows*ncols), &
                  sink_ids(nrows*ncols), sink_dists(nrows*ncols), &
                  is_boundary(nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            return
        end if

        call build_flow_tree_metadata( &
            dirs, valids, x, y, offset_lookup, nrows, ncols, &
            ds_ids, depths, sink_ids, sink_dists, topo_order, topo_cnt, err_code)
        if (err_code /= 0) then
            deallocate (offset_lookup, ds_ids, depths, sink_ids, sink_dists, is_boundary)
            return
        end if

        ! Pre-fill cells on watershed boundaries: max branch distance is the distance to sink
        is_boundary = .false.
        maxbdists = 0.0
        !$omp PARALLEL DO DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, cid, nid, on_border) &
        !$omp COLLAPSE(2) SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                ! ci/cj are loop-controlled and already in bounds, making this
                ! unchecked column-major encoding safe.
                cid = ci + (cj - 1)*nrows
                ! Every stencil offset has magnitude <= 1. Interior cells can
                ! omit all per-neighbour bounds comparisons; only cells on the
                ! thin outer border require explicit coordinate checks.
                on_border = ci == 1 .or. ci == nrows .or. cj == 1 .or. cj == ncols
                do nneighbour = 1, size(boundary_offsets, 1)
                    ni = ci + boundary_offsets(nneighbour, 1)
                    nj = cj + boundary_offsets(nneighbour, 2)
                    if (on_border) then
                        if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    end if
                    if (.not. valids(ni, nj)) cycle
                    ! ni/nj are either interior-safe or checked above, so an
                    ! invalid coordinate cannot wrap into a legitimate ID.
                    nid = ni + (nj - 1)*nrows
                    if (sink_ids(cid) == sink_ids(nid)) cycle
                    is_boundary(cid) = .true.
                    maxbdists(ci, cj) = sink_dists(cid)
                    exit
                end do
            end do
        end do
        !$omp END PARALLEL DO

        ! Reuse the sink such that the watershed is broken down every jump_block_size steps
        do sorder = topo_cnt, 1, -1
            cid = topo_order(sorder)
            nid = ds_ids(cid)
            if (nid == 0 .or. mod(depths(cid), jump_block_size) == 0) then
                ! Sinks and exact block boundaries anchor their own blocks.
                sink_ids(cid) = cid
            else
                ! Other cells share the anchor of their downstream parent.
                sink_ids(cid) = sink_ids(nid)
            end if
        end do
        deallocate (topo_order)

        ! Static scheduling is intentional. Unchunked dynamic scheduling created
        ! one runtime work assignment per collapsed grid cell and dominated the
        ! representative workload.
        !$omp PARALLEL DO DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, cid, nid, on_border) &
        !$omp PRIVATE(conf_id, dist1, dist2) &
        !$omp SCHEDULE(STATIC) COLLAPSE(2)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                cid = ci + (cj - 1)*nrows
                on_border = ci == 1 .or. ci == nrows .or. cj == 1 .or. cj == ncols
                do nneighbour = 1, size(neighbour_offsets, 1)
                    ni = ci + neighbour_offsets(nneighbour, 1)
                    nj = cj + neighbour_offsets(nneighbour, 2)
                    if (on_border) then
                        if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    end if
                    if (.not. valids(ni, nj)) cycle
                    nid = ni + (nj - 1)*nrows
                    ! Different-tree pairs necessarily have both flags set and
                    ! already contributed full sink distances in phase 2.
                    if (is_boundary(cid) .and. is_boundary(nid)) cycle

                    conf_id = find_tree_confluence( &
                                    cid, nid, ds_ids, depths, sink_ids)
                    if (conf_id == 0) then
                        ! Defensive fallback for an invalid/no-confluence query.
                        dist1 = sink_dists(cid)
                        dist2 = sink_dists(nid)
                    else
                        dist1 = sink_dists(cid) - sink_dists(conf_id)
                        dist2 = sink_dists(nid) - sink_dists(conf_id)
                    end if
                    if (.not. is_boundary(cid)) then
                        ! An endpoint participates in several edge updates that
                        ! may be owned by different threads. Atomic MAX prevents
                        ! lost updates without per-thread full-grid result arrays.
                        !$omp ATOMIC UPDATE
                        maxbdists(ci, cj) = max(maxbdists(ci, cj), dist1)
                        !$omp END ATOMIC
                    end if
                    if (.not. is_boundary(nid)) then
                        !$omp ATOMIC UPDATE
                        maxbdists(ni, nj) = max(maxbdists(ni, nj), dist2)
                        !$omp END ATOMIC
                    end if
                end do
            end do
        end do
        !$omp END PARALLEL DO
        deallocate (offset_lookup, ds_ids, depths, sink_ids, sink_dists, is_boundary)
    end subroutine compute_max_branch_dist

    pure subroutine compute_confluence_dist( &
        dists, &
        s1ij, s2ij, dirs, x, y, &
        offset_lookup, check_flag, err_code)
        !! Traces flow paths from two seed cells downstream to compute their confluence distance.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: s1ij(2), s2ij(2)
            !! (i, j) indices of the two seed cells from which to trace flow paths
        integer(c_int8_t), intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical(kind=1), intent(in), optional :: check_flag
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
        logical(kind=1) :: check_flag_
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

        if (present(check_flag)) then
            check_flag_ = check_flag
        else
            check_flag_ = .true.
        end if

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
        implicit none(type, external)
        ! Inputs
        integer, intent(in) :: s1i, s1j, s2i, s2j
            !! Indices of the two seed cells from which to trace flow paths
        integer(c_int8_t), intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, contiguous, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical(kind=1), intent(in), optional :: check_flag
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
        integer :: code1, code2
            !! Unsigned representations of flow direction codes for current cells in paths
        logical(kind=1) :: is_active1, is_active2, local_check_flag
            !! Flags for whether each path is still active (has not reached max length or invalid cell) and local copy of check_flag for performance

        !! Initialisation and checks
        if (present(check_flag)) then
            local_check_flag = check_flag
        else
            local_check_flag = .true.
        end if
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
                code1 = iand(int(dirs(path1(1, npath1), path1(2, npath1))), 255)
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
                code2 = iand(int(dirs(path2(1, npath2), path2(2, npath2))), 255)
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
    !     integer(c_int8_t), dimension(noffsets), intent(in) :: codes
    !     ! Outputs
    !     integer(c_int8_t), dimension(nrows, ncols), intent(out) :: flowdirs

    !     logical, allocatable :: processed(:, :)
    !     integer(c_int8_t), allocatable :: indegs(:, :)
    !     integer, allocatable :: dists(:, :)
    !     integer(c_int8_t), dimension(noffsets) :: opp_codes
    !     integer(c_int8_t) :: noflow_code = 0

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
