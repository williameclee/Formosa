!> Computes raster flow directions using the FORTRAN backend.
!!
!! This internal module is called by the Python drainage API. It 
!! also provides raster-level analyses of the resulting flow field; 
!! flow-graph operations are implemented in the network modules.
!!
!! Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
module drainage_flowdir
    use iso_c_binding, only: c_int8_t
    use utils, only: fill_offset_lookup, find_noflow_code, &
                     array2d_oob, mask2ij
    implicit none(type, external)
contains
    subroutine compute_flowdir_simple( &
        z, valids, dirs, is_flat, nrows, ncols, &
        offsets, codes, noffsets)
        !! Finds D-n flow directions for a given elevation grid,
        !! using the provided flow direction codes and offsets.
        !!
        !! Also identifies flat cells where no flow direction can be
        !! assigned.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
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
        integer(c_int8_t), intent(out) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(out) :: is_flat(nrows, ncols)
            !! Mask indicating which cells are part of flats (i.e.
            !! direction is no-flow)
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
        real :: zmin
            !! (Cell-private) Minimum elevation among valid
            !! neighbours

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
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
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
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
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

    subroutine find_acyclic_flowdirs( &
        dirs, indegs, valids, nrows, ncols, &
        offsets, codes, noffsets, acyclics, err_code)
        !! Identifies valid cells that are not part of a directed
        !! flow cycle.
        !!
        !! Uses Kahn's algorithm to traverse cells from 0-in-degree
        !! seeds, successively removing their outgoing edges. Valid
        !! cells not reached by this traversal belong to a directed
        !! cycle and remain false in 'acyclics'.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        integer(c_int8_t), intent(in) :: indegs(nrows, ncols)
            !! Indegree grid for the valid flow field
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
        logical(kind=1), intent(out) :: acyclics(nrows, ncols)
            !! Mask indicating valid cells removed by Kahn's
            !! algorithm (true for acyclic cells, false otherwise)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Traversal workspace allocation failed
            !!   - 3: Acyclic traversal queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow
            !! direction code, used to find downstream cell indices
        integer(c_int8_t), allocatable :: rem_indegs(:, :)
            !! Remaining indegrees after removing edges from
            !! processed cells
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask of valid zero-indegree cells used to initialise
            !! the queue
        integer, allocatable :: seed_ijs(:, :)
            !! Queue of (i, j) indices awaiting processing
        integer :: alloc_stat
            !! Allocation status code
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and downstream cells
        integer :: iseed, nseeds
            !! Current queue position and final occupied queue
            !! position

        err_code = 0

        allocate (offset_lookup(0:255, 2), &
                  rem_indegs(nrows, ncols), seeds(nrows, ncols), &
                  seed_ijs(2, nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            return
        end if

        seeds = valids .and. (indegs == 0)
        offset_lookup = fill_offset_lookup(offsets, codes)
        call mask2ij( &
            seeds, seed_ijs, size(seed_ijs, dim=2), nseeds, err_code)
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
            if (array2d_oob(ni, nj, nrows, ncols)) cycle
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
end module drainage_flowdir
