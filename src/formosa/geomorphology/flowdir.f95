!!!
! Last modified
!   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
!     - Rename flowdir functions to be more descriptive
!   2026-06-09, En-Chi Lee (williameclee@gmail.com)
!     - Small refactors and documentation cleanup
!     - Renamed function: 'compute_masked_flowdir' -> 'compute_synthetic_flowdir'
!     - Added valids argument to label_flats function
!   2026-06-10, En-Chi Lee (williameclee@gmail.com)
!     - Small refactors and documentation cleanup
!   2026-06-11, En-Chi Lee (williameclee@gmail.com)
!     - Added precomputed 'dist_lookup' for L1 distance in 'compute_dist2source_l1'
!     - Standardised variable, argument, and function names
!!!

module flowdir_utils
    implicit none
contains
    function find_noflow_code(offsets, codes, noffsets, default_noflow_code) result(noflow_code)
        !! For pairs of flow direction codes and their corresponding offsets, find the code that corresponds to the no-flow direction (0, 0). If not found, return the provided default no-flow code or 0 if not provided.
        implicit none
        ! Arguments
        integer, intent(in) :: noffsets
            !! Number of offset codes
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets
        integer*1, intent(in) :: codes(noffsets)
            !! List of codes corresponding to the offsets
        integer*1, intent(in), optional :: default_noflow_code
            !! Optional default no-flow code to use if not found in offsets (default: 0)
        integer*1 :: noflow_code
            !! No-flow code to be returned
        ! Local variables
        integer :: iofs
            !! Offset index for iterating

        ! Assign default no-flow code if not provided
        if (present(default_noflow_code)) then
            noflow_code = default_noflow_code
        else
            noflow_code = 0
        end if

        ! Loop through offsets to find the no-flow code
        do iofs = 1, noffsets
            if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) then
                noflow_code = codes(iofs)
                exit
            end if
        end do
    end function find_noflow_code

    function find_opposite_codes(offsets, codes, noffsets) result(opp_codes)
        !! For pairs of flow direction codes and their corresponding offsets, find the list of codes that correspond to the opposite direction of each code.
        !! For example, if code 1 corresponds to offset (1, 0), and code 2 corresponds to offset (-1, 0), then code 2 is the opposite code of code 1 and vice verse.
        implicit none
        ! Arguments
        integer, intent(in) :: noffsets
            !! Number of offset codes
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets
        integer*1, intent(in) :: codes(noffsets)
            !! List of codes corresponding to the offsets
        integer*1 :: opp_codes(noffsets)
            !! List of opposite codes corresponding to the offsets (same order as input codes)
        ! Local variables
        integer :: iofs, jofs
            !! Offset indices for iterating

        ! Loop through offsets to find opposite codes
        do iofs = 1, noffsets
            do jofs = 1, noffsets
                if (offsets(iofs, 1) == -offsets(jofs, 1) .and. &
                    offsets(iofs, 2) == -offsets(jofs, 2)) then
                    opp_codes(iofs) = codes(jofs)
                    exit
                end if
            end do
        end do
    end function find_opposite_codes

    function fill_offset_lookup(offsets, codes, noffsets) result(diffs)
        !! For pairs of flow direction codes and their corresponding offsets, create a lookup table (array) where the index corresponds to the code and the value is the offset.
        !! The offset codes must be between 0 and 255, and the returned lookup table will have a size of 256-by-2 to accommodate all possible codes. Unused indices will have an offset of (-99, -99) to indicate invalid code.
        !! For example, if code 1 corresponds to offset (1, 0), then diffs(1, :) = (1, 0).
        implicit none
        ! Arguments
        integer, intent(in) :: noffsets
            !! Number of offset codes
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets
        integer*1, intent(in) :: codes(noffsets)
            !! List of codes corresponding to the offsets
        ! Outputs
        integer :: diffs(0:255, 2)
            !! Lookup table for offsets
        ! Local variables
        integer :: iofs

        ! Create lookup tables for offsets
        diffs = -99 ! Initialise to invalid value
        do iofs = 1, noffsets
            if (codes(iofs) < 0 .or. codes(iofs) > 255) then
                print *, "[OFFSET_LOOKUP] Error: Flow direction code out of bounds: ", codes(iofs)
                stop
            end if
            ! Fill in the offset for the corresponding code index
            diffs(codes(iofs), 1) = offsets(iofs, 1)
            diffs(codes(iofs), 2) = offsets(iofs, 2)
        end do
    end function fill_offset_lookup

    subroutine mask2ij( &
        mask, nrows, ncols, ij, nij, cnt)
        !! Converts a 2D logical mask to a list of (i, j) indices where the mask is true.
        !! The output list  will have a maximum size of nij-by-2, and the actual number of valid indices found will be returned in cnt. If the number of valid indices exceeds nij, the remaining will be ignored.
        ! TODO: Optimise this subroutine?
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the input mask
        logical*1, intent(in) :: mask(nrows, ncols)
            !! Input logical mask
        integer, intent(in) :: nij
            !! Maximum number of indices to return
        ! Outputs
        integer, intent(out) :: ij(nij, 2)
            !! Output list of (i, j) indices where mask is true, with a maximum size of nij
        integer, intent(out) :: cnt
            !! Actual number of valid indices found (up to nij)
        ! Local variables
        integer :: ci, cj

        ! Count number of valid neighbors
        cnt = 0

        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. mask(ci, cj)) cycle
                if (cnt == nij) then
                    print *, "Warning: mask2ij found more valid indices than the maximum allowed (", cnt, "). Only the first ", nij, " indices will be returned."
                    return
                end if
                cnt = cnt + 1
                ij(cnt, 1) = ci
                ij(cnt, 2) = cj
            end do
        end do
    end subroutine mask2ij
end module flowdir_utils

module flowdir
    use omp_lib
    use flowdir_utils
    implicit none
contains
    subroutine compute_flowdir_simple( &
        z, valids, dirs, is_flat, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a given elevation grid, using the provided flow direction codes and offsets.
        !! Also identifies flat cells where no flow direction can be assigned.
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
        noflow_code = find_noflow_code(offsets, codes, noffsets)
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
        !! Finds D-n flow directions for a synthetic elevation grid, using the provided flow direction codes and offsets. !! The flow directions are only computed for cells that are part of flats, as indicated by the  label grid. For each flat cell, the flow direction is assigned towards the neighbour with the lowest elevation within the same flat region. If no neighbour has a lower elevation, the cell is assigned the no-flow code.
        !! Note: This function is intended to be used for the synthetic terrain to resolve flats, which should be integer-typed.
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
        noflow_code = find_noflow_code(offsets, codes, noffsets)
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
        !! Finds the cells on the edges of flat areas that drain to lower terrain (low edges) and those that are adjacent to higher terrain (high edges).
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
        noflow_code = find_noflow_code(offsets, codes, noffsets)

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

    subroutine label_flats( &
        z, seeds, valids, flats, nrows, ncols, &
        offsets, noffsets)
        !! Labels connected flat regions in the elevation grid, using a flood-fill algorithm starting from the provided seed cells.
        !! Only valid cells (as indicated by the valids mask) will be considered for labelling. Each flat region will be assigned a unique integer label in the output  grid, while non-flat cells will be assigned 0.
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
        ! Local variables
        integer :: iflat
            !! Index of the current flat region being labeled (!= issed because same flat can have multiple seeds)
        integer, allocatable :: seed_ijs(:, :)
            !! List of (i, j) indices for seed cells
            !! It should be safe to assume that the number of seed cells will not exceed nrows*ncols/2, since each flat region should have at least 2 cells.
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

        allocate (flat_ijs(nrows*ncols, 2))
        allocate (seed_ijs(nrows*ncols/2, 2))
        ! Convert seed mask to list of (i, j) indices
        call mask2ij(seeds, nrows, ncols, &
                     seed_ijs, size(seed_ijs, dim=1), nseeds)

        flats = 0
        iflat = 1
        iseed = 1
        ! Loop over seed cells to label flats using a flood-fill algorithm
        do iseed = 1, nseeds
            si = seed_ijs(iseed, 1)
            sj = seed_ijs(iseed, 2)

            ! Skip if not valid
            if (.not. valids(si, sj)) cycle
            ! Skip if already labeled
            if (flats(si, sj) /= 0) cycle

            sz = z(si, sj)

            ! Reset buffer
            ifill = 1
            nfills = 1
            flat_ijs(ifill, :) = [si, sj]
            flats(si, sj) = iflat

            do while (ifill <= nfills)
                ci = flat_ijs(ifill, 1)
                cj = flat_ijs(ifill, 2)
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
                    if (nfills > size(flat_ijs, 1)) then
                 print *, "[LABEL_FLAT] Error: Flat flooding buffer overflow (size:", nfills, ", allocated:", size(flat_ijs, 1), ")"
                        stop
                    end if
                    flat_ijs(nfills, :) = [ni, nj]
                    flats(ni, nj) = iflat
                end do

            end do

            iflat = iflat + 1
        end do
        deallocate (flat_ijs)
        deallocate (seed_ijs)
    end subroutine label_flats

    subroutine create_pushing_syn_grad( &
        z, flats, nrows, ncols, &
        high_edges, offsets, noffsets)
        !! Produces a synthetic elevation that decreases away from 'high edges' of flats.
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
        ! Local variables
        integer :: nflats
            !! Number of unique flat labels (excluding 0 for non-flat cells)
        integer :: dist
            !! Current distance from high edges, used to assign synthetic elevation values
        integer, allocatable :: maxdist(:)
            !! Maximum synthetic elevation value assigned to each flat region, used to adjust final z values to ensure they flow away from high edges
        integer :: iedge, nedges
            !! Index for iterating through high edge cells and total number of high edge cells in the queue
            !! As the algorithm proceeds, new cells will be added to the queue and nedges will be updated accordingly
        integer :: iofs
            !! Index for iterating through offsets
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        integer, parameter :: marker(2) = [-1, -1]
            !! Special index used to mark the end of each iteration in the queue
        logical*1 :: added_since_marker
            !! Flag to track whether new cells have been added to the queue since the last marker, used to determine when to stop the algorithm
        logical*1, allocatable :: queued(:, :)
            !! Mask to track which cells have already been added to the queue, to avoid adding the same cell multiple times
        integer, allocatable :: high_edge_ijs(:, :)
            !! List of (i, j) indices for high edge cells to be processed in the algorithm, used as a queue for breadth-first search
        integer :: max_queue_size
            !! Maximum size of the queue buffer for high edge cells ('high_edges_ijs', including the marker)

        max_queue_size = count(flats /= 0) + max(nrows, ncols)*(maxval(flats) - minval(flats) + 1)
        allocate (high_edge_ijs(max_queue_size, 2))

        high_edge_ijs = 0
        nedges = 0
        z = 0
        call mask2ij(high_edges, nrows, ncols, &
                     high_edge_ijs, size(high_edge_ijs, dim=1), nedges)
        if (nedges == 0) then
            ! No high edges found, set z to zero and exit
            deallocate (high_edge_ijs)
            return
        end if

        nedges = nedges + 1
        high_edge_ijs(nedges, :) = marker

        nflats = maxval(flats)
        allocate (maxdist(nflats))
        maxdist = 0

        allocate (queued(nrows, ncols))
        queued = .false.
        added_since_marker = .false.

        ! Mark initial seeds as queued
        do iedge = 1, nedges - 1
            ci = high_edge_ijs(iedge, 1)
            cj = high_edge_ijs(iedge, 2)
            queued(ci, cj) = .true.
        end do
        ! Loop through all high_edges to find cells flowing away from flats
        ! After this the first loop, z values decreases towards high edges (opposite of desired)
        dist = 1
        iedge = 1
        do while (iedge <= nedges)
            ci = high_edge_ijs(iedge, 1)
            cj = high_edge_ijs(iedge, 2)
            iedge = iedge + 1

            ! Check for marker to separate iterations
            if (ci == marker(1) .and. cj == marker(2)) then
                ! Break if no more cells to process
                if (.not. added_since_marker) exit
                ! Skip if encountered marker
                dist = dist + 1
                nedges = nedges + 1
                ! Check buffer size
                if (nedges > max_queue_size) then
                   print *, "[AWAY_FROM_HIGH] Error: High edges buffer overflow (size:", nedges, ", allocated:", max_queue_size, ")"
                    stop
                end if
                high_edge_ijs(nedges, :) = marker
                added_since_marker = .false.
                cycle
            end if

            ! Check bounds after marker check
            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
                print *, "[AWAY_FROM_HIGH] Error: Current index out of bounds (", ci, ",", cj, ")"
                stop
            else if (flats(ci, cj) == 0) then
                ! Skip if for some reason we ended up with a non-flat cell in the queue
                print *, "[AWAY_FROM_HIGH] Warning: Encountered non-flat cell in queue at (", ci, ",", cj, "). This should not happen, but will be skipped."
                cycle
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
                   print *, "[AWAY_FROM_HIGH] Error: High edges buffer overflow (size:", nedges, ", allocated:", max_queue_size, ")"
                    stop
                end if
                high_edge_ijs(nedges, :) = [ni, nj]
                queued(ni, nj) = .true.
                added_since_marker = .true.
            end do
        end do
        deallocate (high_edge_ijs)
        deallocate (queued)

        ! Adjust z values within flats to ensure they flow away from high edges
        do concurrent(ci=1:nrows, cj=1:ncols, flats(ci, cj) /= 0)
            z(ci, cj) = maxdist(flats(ci, cj)) - z(ci, cj) + 1
        end do
        deallocate (maxdist)
    end subroutine create_pushing_syn_grad

    subroutine create_pulling_syn_grad( &
        z, flats, nrows, ncols, &
        low_edges, offsets, noffsets)
        !! Produces a synthetic elevation that drains towards 'low edges' of flats.
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
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer, parameter :: marker(2) = [-1, -1]
            !! Special marker to indicate the end of an iteration in the queue
        logical*1 :: added_since_marker
            !! Flag to track whether new cells have been added to the queue since the last marker, used to determine when to stop the algorithm
        integer :: iedge, nedges
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
            !! Maximum size of the queue buffer for low edge cells ('low_edges_ijs', including the marker)

        max_queue_size = count(flats /= 0) + max(nrows, ncols)*maxval(flats)
        allocate (low_edges_ijs(max_queue_size, 2))
        call mask2ij(low_edges, nrows, ncols, &
                     low_edges_ijs, size(low_edges_ijs, dim=1), nedges)
        nedges = nedges + 1
        low_edges_ijs(nedges, :) = marker

        ! Initialise z to zero
        z = 0
        allocate (queued(nrows, ncols))
        queued = .false.

        ! Mark initial seeds as queued
        do iedge = 1, nedges - 1
            ci = low_edges_ijs(iedge, 1)
            cj = low_edges_ijs(iedge, 2)
            queued(ci, cj) = .true.
        end do

        ! Loop through all low_edges to find cells flowing into flats
        iedge = 1
        dist = 1
        added_since_marker = .false.
        do while (iedge <= nedges)
            ci = low_edges_ijs(iedge, 1)
            cj = low_edges_ijs(iedge, 2)
            iedge = iedge + 1

            if (ci == marker(1) .and. cj == marker(2)) then
                ! Break if no more cells to process
                if (.not. added_since_marker) exit
                ! Skip if encountered marker
                dist = dist + 1
                nedges = nedges + 1
                ! Check buffer size
                if (nedges > max_queue_size) then
                    print *, "[TOWARDS_LOW] Error: Low edges buffer overflow (size:", nedges, ", allocated:", max_queue_size, ")"
                    stop
                end if
                low_edges_ijs(nedges, :) = marker
                added_since_marker = .false.
                cycle
            end if

            ! Check bounds after marker check
            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
                print *, "[TOWARDS_LOW] Error: Current indices out of bounds (", ci, ",", cj, ")"
                stop
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
                    print *, "[TOWARDS_LOW] Error: Low edges buffer overflow (size:", nedges, ", allocated:", max_queue_size, ")"
                    stop
                end if
                low_edges_ijs(nedges, :) = [ni, nj]
                queued(ni, nj) = .true.
                added_since_marker = .true.
            end do
        end do
        deallocate (queued)
        deallocate (low_edges_ijs)
    end subroutine create_pulling_syn_grad

    subroutine count_indegree( &
        dirs, indegs, nrows, ncols, &
        offsets, codes, noffsets)
        !! Computes the number of upstream cells (indegs) for each cell in a flow direction grid.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
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

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                ! Loop over offsets to find neighbours flowing into current cell
                do iofs = 1, noffsets
                    ! Upstream neighbour indices
                    ni = ci - offsets(iofs, 1)
                    nj = cj - offsets(iofs, 2)
                    ! Check bounds
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
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
        offsets, codes, noffsets)
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
        allocate (offset_lookup(0:255, 2))
        offset_lookup = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero in-degrees
        max_queue_size = nrows*ncols
        allocate (flood_ijs(max_queue_size, 2))
        allocate (flood_seeds(nrows, ncols))
        flood_seeds = valids .and. (indegs == 0)
        call mask2ij(flood_seeds, nrows, ncols, &
                     flood_ijs, max_queue_size, ntofills)
        deallocate (flood_seeds)

        accumulations = areas
        itofill = 1
        do while (itofill <= ntofills)
            ci = flood_ijs(itofill, 1)
            cj = flood_ijs(itofill, 2)
            itofill = itofill + 1

            ni = ci + offset_lookup(dirs(ci, cj), 1)
            nj = cj + offset_lookup(dirs(ci, cj), 2)

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
                print *, "[FLOW_ACCUMULATION] Error: Flooding buffer overflow (size:", ntofills, ", allocated:", max_queue_size, ")"
                stop
            end if
            flood_ijs(ntofills, :) = [ni, nj]
        end do
        deallocate (offset_lookup)
        deallocate (flood_ijs)
    end subroutine compute_flow_accumulation

    subroutine compute_dist2source_l1( &
        dirs, valids, indegs, dists, nrows, ncols, &
        offsets, codes, noffsets)
        !! Computes the distance to the nearest source cell (cell with zero indegree) for each cell in a flow direction grid, using a breadth-first search starting from source cells.
        !! The distance in measured in the number of cells along the flow path (i.e. integer-typed L1 distance).
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
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer, allocatable :: dist_lookup(:)
            !! Lookup table for distance increments corresponding to each flow direction code, used to calculate distance to downstream cells
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
        allocate (offset_lookup(0:255, 2))
        allocate (dist_lookup(0:255))
        offset_lookup = fill_offset_lookup(offsets, codes, noffsets)
        dist_lookup = sum(abs(offset_lookup), dim=2)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(max_queue_size, 2))
        allocate (tofill_seeds(nrows, ncols))
        tofill_seeds = valids .and. (indegs == 0)
        call mask2ij(tofill_seeds, nrows, ncols, &
                     tofill_ijs, max_queue_size, ntofills)
        deallocate (tofill_seeds)

        !! Main loop to fill distances using a breadth-first search starting from source cells
        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(itofill, 1)
            cj = tofill_ijs(itofill, 2)
            itofill = itofill + 1

            ni = ci + offset_lookup(dirs(ci, cj), 1)
            nj = cj + offset_lookup(dirs(ci, cj), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            if (dists(ci, cj) + 1 > dists(ni, nj)) then
                dists(ni, nj) = dists(ci, cj) + dist_lookup(dirs(ci, cj))
            end if
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > max_queue_size) then
                    print *, "[DIST2SOURCE_L1] Error: tofill buffer overflow (size:", ntofills, ", allocated:", max_queue_size, ")"
                    stop
                end if
                tofill_ijs(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (offset_lookup)
        deallocate (dist_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_dist2source_l1

    subroutine compute_dist2source( &
        dirs, valids, x, y, indegs, dists, nrows, ncols, &
        offsets, codes, noffsets)
        !! Computes the distance downstream along flow directions for each cell in the flow direction grid.
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
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        real :: step_dist
            !! Distance between current cell and downstream cell, calculated using x and y coordinates
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the flooding algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the flooding buffer ('tofill_ijs')

        ! Create lookup tables for offsets
        allocate (offset_lookup(0:255, 2))
        offset_lookup = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(max_queue_size, 2))
        allocate (seeds(nrows, ncols))
        seeds = valids .and. (indegs == 0)
        call mask2ij(seeds, nrows, ncols, &
                     tofill_ijs, max_queue_size, ntofills)
        deallocate (seeds)

        !! Main loop to fill distances using a breadth-first search starting from source cells
        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(itofill, 1)
            cj = tofill_ijs(itofill, 2)
            itofill = itofill + 1

            ni = ci + offset_lookup(dirs(ci, cj), 1)
            nj = cj + offset_lookup(dirs(ci, cj), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            step_dist = hypot( &
                        x(ni, nj) - x(ci, cj), &
                        y(ni, nj) - y(ci, cj))
            if (dists(ci, cj) + step_dist > dists(ni, nj)) then
                dists(ni, nj) = dists(ci, cj) + step_dist
            end if
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > max_queue_size) then
                    print *, "[DIST2SOURCE] Error: tofill buffer overflow (size:", ntofills, ", allocated:", max_queue_size, ")"
                    stop
                end if
                tofill_ijs(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (offset_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_dist2source

    subroutine compute_dist2sink( &
        dists, dirs, x, y, valids, nrows, ncols, offsets, codes, noffsets)
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

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        dists = -1

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(max_queue_size, 2))
        allocate (seeds(nrows, ncols))
        seeds = valids .and. (dirs == noflow_code)
        call mask2ij(seeds, nrows, ncols, &
                     seed_ijs, max_queue_size, nseeds)
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs)
        allocate (tofill_ijs(max_queue_size, 2))
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            si = seed_ijs(iseed, 1)
            sj = seed_ijs(iseed, 2)

            ! Loop through buffer
            nfills = 1
            ifill = 1
            dists(si, sj) = 0.0
            tofill_ijs(1, :) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(ifill, 1)
                cj = tofill_ijs(ifill, 2)
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
                        print *, "[DIST2SINK] Error: tofill buffer overflow (size:", nfills, ", allocated:", max_queue_size, ")"
                        stop
                    end if
                    tofill_ijs(nfills, :) = [ui, uj]
                    ! Compute distance
                    dists(ui, uj) = dists(ci, cj) + hypot( &
                                    x(ui, uj) - x(ci, cj), &
                                    y(ui, uj) - y(ci, cj))
                end do
            end do
        end do
        !$omp END DO
        deallocate (tofill_ijs)
        !$omp END PARALLEL
        deallocate (seed_ijs)
    end subroutine compute_dist2sink

    subroutine compute_flow_strahler_order( &
        dirs, valids, indegs, orders, nrows, ncols, &
        offsets, codes, noffsets)
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
            !! Grid of Strahler stream order values for each cell, where source cells (cells with zero indegree) have order 1, and the order of downstream cells is determined by the orders of their upstream cells according to Strahler's rules.
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical*1, allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be processed in the breadth-first search from source cells
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed ('tofill_ijs')

        ! Create lookup tables for offsets
        allocate (offset_lookup(0:255, 2))
        offset_lookup = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero indegree
        max_queue_size = nrows*ncols
        allocate (tofill_ijs(max_queue_size, 2))
        allocate (seeds(nrows, ncols))
        seeds = valids .and. (indegs == 0)
        call mask2ij(seeds, nrows, ncols, &
                     tofill_ijs, max_queue_size, ntofills)
        deallocate (seeds)

        orders = 1
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(itofill, 1)
            cj = tofill_ijs(itofill, 2)
            itofill = itofill + 1

            ni = ci + offset_lookup(dirs(ci, cj), 1)
            nj = cj + offset_lookup(dirs(ci, cj), 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            if (orders(ni, nj) < orders(ci, cj)) then
                orders(ni, nj) = orders(ci, cj)
            else if (orders(ni, nj) == orders(ci, cj)) then
                orders(ni, nj) = orders(ni, nj) + int(1, kind=2)
            end if
            ! Decrement indegree of downstream cell
            indegs(ni, nj) = indegs(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegs(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > max_queue_size) then
             print *, "[COMPUTE_STRAHLER_ORDER] Error: tofill buffer overflow (size:", ntofills, ", allocated:", max_queue_size, ")"
                    stop
                end if
                tofill_ijs(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (offset_lookup)
        deallocate (tofill_ijs)
    end subroutine compute_flow_strahler_order

    subroutine label_watersheds( &
        labels, dirs, valids, nrows, ncols, offsets, codes, noffsets)
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

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        labels = 0

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(max_queue_size, 2))
        allocate (seeds(nrows, ncols))
        seeds = valids .and. (dirs == noflow_code)
        call mask2ij(seeds, nrows, ncols, &
                     seed_ijs, max_queue_size, nseeds)
        deallocate (seeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs)
        allocate (tofill_ijs(max_queue_size, 2))
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            si = seed_ijs(iseed, 1)
            sj = seed_ijs(iseed, 2)

            ! Loop through buffer
            nfills = 1
            ifill = 1
            labels(si, sj) = iseed
            tofill_ijs(1, :) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(ifill, 1)
                cj = tofill_ijs(ifill, 2)
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
                    print *, "[LABEL_WATERSHEDS] Error: To-fill buffer overflow (size:", nfills, ", allocated:", max_queue_size, ")"
                        stop
                    end if
                    tofill_ijs(nfills, :) = [ui, uj]
                    ! Compute distance
                    labels(ui, uj) = labels(ci, cj)
                end do
            end do
        end do
        !$omp END DO
        deallocate (tofill_ijs)
        !$omp END PARALLEL
    end subroutine label_watersheds

    subroutine flood_upstream( &
        flooded, dirs, seeds, valids, nrows, ncols, offsets, codes, noffsets)
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

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        flooded = .false.

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ijs(max_queue_size, 2))
        call mask2ij(seeds, nrows, ncols, &
                     seed_ijs, max_queue_size, nseeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_ijs)
        allocate (tofill_ijs(max_queue_size, 2))
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            si = seed_ijs(iseed, 1)
            sj = seed_ijs(iseed, 2)

            ! Check if is valid
            if (.not. valids(si, sj)) cycle

            ! Loop through buffer
            nfills = 1
            ifill = 1
            flooded(si, sj) = .true.
            tofill_ijs(1, :) = [si, sj]

            do while (ifill <= nfills)
                ci = tofill_ijs(ifill, 1)
                cj = tofill_ijs(ifill, 2)
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
                       print *, "[FLOOD_UPSTREAM] Error: tofill buffer overflow (size:", nfills, ", allocated:", max_queue_size, ")"
                        stop
                    end if
                    tofill_ijs(nfills, :) = [ui, uj]
                    ! Compute distance
                    flooded(ui, uj) = .true.
                end do
            end do
        end do
        !$omp END DO
        deallocate (seed_ijs)
        deallocate (tofill_ijs)
        !$omp END PARALLEL
    end subroutine flood_upstream

    subroutine compute_max_branch_dist( &
        maxbdists, dirs, valids, x, y, basin_ids, nrows, ncols, &
        offsets, codes, noffsets)
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
        ! Local variables
        integer, allocatable :: diffs(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        real :: dists(2)
            !! Array to hold distances from two neighbouring cells to their confluence point
        integer :: nneighbour, neighbour_offsets(4, 2)
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        integer :: maxlen, path1id, path2id
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)
        logical*1, allocatable :: is_max_dist(:, :)

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

        ! Define neighbour offsets
        neighbour_offsets = &
            reshape([1, -1, & ! SW
                     0, 1, & ! E
                     1, 1, & ! SE
                     1, 0 & ! S
                     ], [4, 2])

        maxlen = 2*(nrows + ncols)

        allocate (is_max_dist(nrows, ncols))
        maxbdists = 0.0
        is_max_dist = .false.
        !$omp PARALLEL DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, dists) &
        !$omp PRIVATE(path1, path2, path1id, path2id, visited)
        allocate (path1(maxlen, 2))
        allocate (path2(maxlen, 2))
        allocate (visited(nrows, ncols))
        visited = 0
        path1id = 1
        path2id = 2
        !$omp DO SCHEDULE(DYNAMIC) &
        !$omp COLLAPSE(2)
        do cj = 1, ncols
            do ci = 1, nrows
                do nneighbour = 1, size(neighbour_offsets, 1)
                    if (.not. valids(ci, cj)) cycle
                    ni = ci + neighbour_offsets(nneighbour, 1)
                    nj = cj + neighbour_offsets(nneighbour, 2)
                    ! Check bounds
                    if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                    ! Check mask
                    if (.not. valids(ni, nj)) cycle
                    if (is_max_dist(ci, cj) .and. is_max_dist(ni, nj)) cycle
                    call inner_compute_confluence_dist( &
                        dists, ci, cj, ni, nj, dirs, x, y, diffs, &
                        maxpathlen=maxlen, path1=path1, path2=path2, &
                        visited=visited, id1=path1id, id2=path2id, &
                        check_flag=logical(basin_ids(ni, nj) == basin_ids(ci, cj), kind=1))
                    maxbdists(ci, cj) = max(maxbdists(ci, cj), dists(1))
                    !$omp ATOMIC UPDATE
                    maxbdists(ni, nj) = max(maxbdists(ni, nj), dists(2))
                    !$omp END ATOMIC

                    ! If different basin ids, mark as max distance computed
                    if (basin_ids(ni, nj) /= basin_ids(ci, cj)) then
                        is_max_dist(ci, cj) = .true.
                        is_max_dist(ni, nj) = .true.
                    end if

                    if (path1id > 2147483640) then
                        visited = 0
                        path1id = 1
                        path2id = 2
                    end if
                    path1id = path1id + 2
                    path2id = path2id + 2
                end do
            end do
        end do
        !$omp END DO
        deallocate (path1)
        deallocate (path2)
        deallocate (visited)
        !$omp END PARALLEL
        deallocate (is_max_dist)
        deallocate (diffs)
    end subroutine compute_max_branch_dist

    subroutine compute_confluence_dist( &
        dists, &
        s1ij, s2ij, dirs, x, y, &
        offset_lookup, check_flag)
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
        ! Local variables
        logical*1 :: check_flag_
        integer :: maxpathlen
        integer :: id1, id2
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)

        maxpathlen = 4*(size(dirs, 1) + size(dirs, 2))
        id1 = 1
        id2 = 2
        allocate (path1(maxpathlen, 2))
        allocate (path2(maxpathlen, 2))
        allocate (visited(size(dirs, 1), size(dirs, 2)))
        visited = 0

        check_flag_ = (.not. present(check_flag)) .or. check_flag

        call inner_compute_confluence_dist( &
            dists, &
            s1ij(1), s1ij(2), s2ij(1), s2ij(2), dirs, x, y, offset_lookup, &
            maxpathlen, path1, path2, visited, id1, id2, &
            check_flag=check_flag_)
        deallocate (path1)
        deallocate (path2)
        deallocate (visited)
    end subroutine compute_confluence_dist

    subroutine inner_compute_confluence_dist( &
        dists, s1i, s1j, s2i, s2j, dirs, x, y, &
        offset_lookup, maxpathlen, path1, path2, visited, id1, id2, check_flag)
        implicit none
        ! Inputs
        integer, intent(in) :: s1i, s1j, s2i, s2j
            !! Indices of the two seed cells from which to trace flow paths
        integer*1, intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical*1, intent(in), optional :: check_flag
            !! Flag for whether to check for confluence at each step (can be turned off for performance if many confluences are expected)
        integer, intent(in) :: maxpathlen
            !! Maximum path length to search before giving up and assuming no confluence (should be large enough to allow confluence but prevent infinite loops in case of errors)
        integer, intent(inout) :: path1(maxpathlen, 2), path2(maxpathlen, 2)
            !! Workspace arrays for paths and visited grid (to avoid repeated allocation in recursive calls)
        integer, intent(inout) :: visited(:, :)
            !! Grid to track visited paths by ids
        ! Outputs
        real :: dists(2)
            !! Distances from each seed ceel to the confluence cell (or to max path length if no confluence found)
        ! Local variables
        integer :: id1, id2
            !! Unique ids to mark visited cells for each path in the visited grid
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
        is_active1 = .true.
        is_active2 = .true.

        ! Return zero if same cell
        if ((s1i == s2i) .and. (s1j == s2j)) then
            dists = 0.0
            return
        end if

        !! Main algorithm
        npath1 = 1
        path1(npath1, 1) = s1i
        path1(npath1, 2) = s1j
        visited(s1i, s1j) = id1
        npath2 = 1
        path2(npath2, 1) = s2i
        path2(npath2, 2) = s2j
        visited(s2i, s2j) = id2

        tracer_loop: do while (is_active1 .or. is_active2)
            path1_prc: block
                if (.not. is_active1) exit path1_prc
                ! Make sure code is valid
                code1 = dirs(path1(npath1, 1), path1(npath1, 2))
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
                n1i = path1(npath1, 1) + offset_lookup(code1, 1)
                n1j = path1(npath1, 2) + offset_lookup(code1, 2)
                if (n1i < 1 .or. n1i > size(dirs, 1) .or. n1j < 1 .or. n1j > size(dirs, 2)) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                else if (npath1 >= maxpathlen) then
                    print *, "[CONFLUENCE_DISTANCE] Warning: Path 1 exceeded max length of ", maxpathlen
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                end if
                npath1 = npath1 + 1
                path1(npath1, :) = [n1i, n1j]
                ! Check for self-intersection
                if (visited(n1i, n1j) == id1) then
                    print *, "[CONFLUENCE_DISTANCE] Warning: Path 1 self-intersection at ", n1i, ",", n1j
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path1_prc
                if (visited(n1i, n1j) /= id2) then
                    visited(n1i, n1j) = id1
                    exit path1_prc
                end if
                ! Confluence found
                do ipath2 = 1, npath2
                    if (.not. all(path2(ipath2, :) == [n1i, n1j])) cycle
                    iconf1 = npath1
                    iconf2 = ipath2
                    exit tracer_loop
                    if (ipath2 < npath2) cycle
                    print *, "[CONFLUENCE_DISTANCE] Error: Confluence promised but not found"
                    iconf1 = npath1
                end do
            end block path1_prc

            path2_prc: block
                if (.not. is_active2) exit path2_prc
                ! Make sure code is valid
                code2 = dirs(path2(npath2, 1), path2(npath2, 2))
                if (code2 < lbound(offset_lookup, 1) .or. code2 > ubound(offset_lookup, 1)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (offset_lookup(code2, 1) == 0 .and. offset_lookup(code2, 2) == 0) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                end if
                n2i = path2(npath2, 1) + offset_lookup(code2, 1)
                n2j = path2(npath2, 2) + offset_lookup(code2, 2)
                if (n2i < 1 .or. n2i > size(dirs, 1) .or. n2j < 1 .or. n2j > size(dirs, 2)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (npath2 >= maxpathlen) then
                    print *, "[CONFLUENCE_DISTANCE] Warning: Path 2 exceeded max length of ", maxpathlen
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                end if
                npath2 = npath2 + 1
                path2(npath2, :) = [n2i, n2j]
                ! Check for self-intersection
                if (visited(n2i, n2j) == id2) then
                    print *, "[CONFLUENCE_DISTANCE] Warning: Path 2 self-intersection at ", n2i, ",", n2j
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path2_prc
                if (visited(n2i, n2j) /= id1) then
                    visited(n2i, n2j) = id2
                    exit path2_prc
                end if
                ! Confluence found
                do ipath1 = 1, npath1
                    if (.not. all(path1(ipath1, :) == [n2i, n2j])) cycle
                    iconf1 = ipath1
                    iconf2 = npath2
                    exit tracer_loop
                    if (ipath1 < npath1) cycle
                    print *, "[CONFLUENCE_DISTANCE] Error: Confluence promised but not found"
                    iconf2 = npath2
                end do
            end block path2_prc
        end do tracer_loop

        ! Compute distances to confluence
        do ipath1 = 1, min(iconf1, npath1) - 1
            dists(1) = dists(1) + hypot( &
                       x(path1(ipath1 + 1, 1), path1(ipath1 + 1, 2)) - x(path1(ipath1, 1), path1(ipath1, 2)), &
                       y(path1(ipath1 + 1, 1), path1(ipath1 + 1, 2)) - y(path1(ipath1, 1), path1(ipath1, 2)))
        end do
        do ipath2 = 1, min(iconf2, npath2) - 1
            dists(2) = dists(2) + hypot( &
                       x(path2(ipath2 + 1, 1), path2(ipath2 + 1, 2)) - x(path2(ipath2, 1), path2(ipath2, 2)), &
                       y(path2(ipath2 + 1, 1), path2(ipath2 + 1, 2)) - y(path2(ipath2, 1), path2(ipath2, 2)))
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

    !     noflow_code = find_noflow_code(offsets, codes, noffsets)
    !     opp_codes = find_opposite_codes(offsets, codes, noffsets)

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
end module flowdir
