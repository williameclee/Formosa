!!!
! Calculations of raster cell-level geomorphological metrics of a
! digital elevation model (DEM) raster based on the flow direction.
!
! Content of this file is mostly designed to be called by the Python
! frontend and not directly by the user.
!
! Last modified: 2026-08-05, En-Chi Lee (williameclee@gmail.com)
!!!

module drainage_metrics
    use iso_c_binding, only: c_int8_t, c_int16_t
    use utils, only: fill_offset_lookup, find_noflow_code, &
                     array2d_oob, mask2id, mask2ij, &
                     id2ij_checked, ij2id_checked
    use distances, only: l1dist_xy, l2dist_xy
    implicit none(type, external)
contains
    subroutine compute_flow_accumulation( &
        dirs, valids, areas, indegs, accums, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes flow accumulation for each cell in a flow
        !! direction grid.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        real, intent(in) :: areas(nrows, ncols)
            !! Area of each cell, used as the initial accumulation
            !! value for each cell
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that
            !! flow into each cell.
            !! This will be modified in-place during the algorithm
            !! to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
        ! Outputs
        real, intent(out) :: accums(nrows, ncols)
            !! Grid of flow accumulation values, i.e. total area
            !! flowing into each cell
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Flooding queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow
            !! direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total
            !! number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        integer, allocatable :: flood_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be
            !! processed in the flooding algorithm
        integer :: max_queue_size
            !! Maximum size of the flooding buffer ('flood_ijs')
        logical(kind=1), allocatable :: flood_seeds(:, :)
            !! Mask to identify initial seed cells for the flooding
            !! algorithm (valid cells with zero in-degrees)

        ! Guard nrows*ncols before using it as a default-integer
        ! allocation extent or in the column-major linear-index
        ! expressions below.
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if

        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill the tofill buffer with all valid cells with zero
        ! in-degrees
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
        call mask2ij( &
            flood_seeds, flood_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (flood_seeds)

        err_code = 0
        accums = areas
        itofill = 1
        do while (itofill <= ntofills)
            ci = flood_ijs(1, itofill)
            cj = flood_ijs(2, itofill)
            itofill = itofill + 1

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (array2d_oob(ni, nj, nrows, ncols)) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update accumulation of downstream cell
            accums(ni, nj) = accums(ni, nj) + accums(ci, cj)
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
        !! Computes the distance to the nearest source cell (cell
        !! with zero in-degree) for each cell in a flow direction
        !! grid, using a breadth-first search starting from source
        !! cells.
        !!
        !! The distance in measured in the number of cells along the
        !! flow path (i.e. integer-typed L1 distance).
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that
            !! flow into each cell
            !! This will be modified in-place during the algorithm
            !! to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
        ! Outputs
        integer, intent(out) :: dists(nrows, ncols)
            !! Grid of distances to the nearest source cell (cell
            !! with zero in-degree).
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Source-distance queue capacity was exceeded
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow
            !! direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total
            !! number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical(kind=1), allocatable :: tofill_seeds(:, :)
            !! Mask to identify initial seed cells for the flooding
            !! algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be
            !! processed in the flooding algorithm
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

        ! Fill tofill buffer with all valid cells with 0 in-degree
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
        call mask2ij( &
            tofill_seeds, tofill_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (tofill_seeds)

        ! Main loop to fill distances using a breadth-first search
        ! starting from source cells
        err_code = 0
        dists = 0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_ijs(1, itofill)
            cj = tofill_ijs(2, itofill)
            itofill = itofill + 1

            ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
            nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

            ! Check bounds
            if (array2d_oob(ni, nj, nrows, ncols)) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            dists(ni, nj) = &
                max(dists(ci, cj), dists(ci, cj) + l1dist_xy(ni, nj, ci, cj))
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
        !! Computes the distance downstream along flow directions
        !! for each cell in the flow direction grid.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to
            !! calculate distances between cells
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that
            !! flow into each cell
            !! This will be modified in-place during the algorithm
            !! to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
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
            !! Lookup table for offsets corresponding to each flow
            !! direction code, used to find downstream cell indices
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total
            !! number of cells to fill
        integer :: ci, cj, ni, nj
            !! Rows/columns for current and neighbour cells
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the flooding
            !! algorithm (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be
            !! processed in the flooding algorithm
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
        call mask2ij( &
            seeds, tofill_ijs, max_queue_size, ntofills, err_code)
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
            if (array2d_oob(ni, nj, nrows, ncols)) cycle
            ! Check mask
            if (.not. valids(ni, nj)) cycle
            ! Check not a self-loop
            if (ni == ci .and. nj == cj) cycle
            ! Check not already processed
            if (indegs(ni, nj) <= 0) cycle

            ! Update distance of downstream cell
            dists(ni, nj) = &
                max(dists(ci, cj), &
                    dists(ci, cj) + l2dist_xy(x(ni, nj), y(ni, nj), x(ci, cj), y(ci, cj)))
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
        dists, dirs, x, y, valids, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes the distance upstream along flow directions for
        !! each cell in the flow direction grid.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction grid encoded using codes.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate flow-edge
            !! distances
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer, intent(in) :: noffsets
            !! Number of supported flow-direction codes
        integer, intent(in) :: offsets(noffsets, 2)
            !! Row/column displacement corresponding to each
            !! direction code
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! External direction codes corresponding to offsets
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
            !! Code corresponding to noflow direction, used to
            !! identify sink cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer,
            !! and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream cells
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid
            !! cells with noflow direction)
        integer, allocatable :: seed_ids(:)
            !! Buffer for storing linear IDs of seed cells to be
            !! processed in the algorithm
        integer, allocatable :: tofill_ids(:)
            !! Buffer for storing linear cell IDs in the breadth-
            !! first search from sink cells
        integer :: cell_id
        logical(kind=1) :: id_is_valid
        integer :: max_queue_size
            !! Maximum number of linear cell IDs in the seed and
            !! traversal queues
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
        call mask2id( &
            seeds, seed_ids, max_queue_size, nseeds, err_code)
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
            call id2ij_checked( &
                seed_ids(iseed), nrows, ncols, si, sj, id_is_valid)
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
                call id2ij_checked( &
                    tofill_ids(ifill), nrows, ncols, ci, cj, id_is_valid)
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
                    if (array2d_oob(ui, uj, nrows, ncols)) cycle
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
                    dists(ui, uj) = &
                        dists(ci, cj) + l2dist_xy(x(ui, uj), y(ui, uj), x(ci, cj), y(ci, cj))
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
            !! Validity mask (false for no-data)
        integer(c_int8_t), intent(inout) :: indegs(nrows, ncols)
            !! Indegree grid, i.e. number of upstream cells that
            !! flow into each cell
            !! This will be modified in-place during the algorithm
            !! to track which cells have been processed.
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
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
            !! Lookup table for offsets corresponding to each flow
            !! direction code, used to find downstream cell indices
        integer :: iofs
            !! Index for iterating through offsets
        integer :: itofill, ntofills
            !! Index for iterating through cells to fill and total
            !! number of cells to fill
        integer :: ci, cj, ni, nj, ui, uj
            !! Rows/columns for current and neighbour downstream and
            !! upstream cells
        integer(c_int16_t) :: max_uorder
            !! Maximum Strahler stream order value of a cell's
            !! upstream neighbours
        logical(kind=1) :: increase_order
            !! Whether the current cell's order should be increased
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify initial seed cells for the algorithm
            !! (valid cells with zero indegree)
        integer, allocatable :: tofill_ijs(:, :)
            !! Buffer for storing (i, j) indices of cells to be
            !! processed in the breadth-first search from source
            !! cells
        integer :: max_queue_size
            !! Maximum size of the buffer for cells to be processed
            !! ('tofill_ijs')

        ! Create lookup tables for offsets
        err_code = 0
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Fill tofill buffer with all valid cells with 0 in-degree
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
        call mask2ij( &
            seeds, tofill_ijs, max_queue_size, ntofills, err_code)
        if (err_code /= 0) return
        deallocate (seeds)

        itofill = 1

        ! Push all the seeds' downstream cells to the queue
        do while (itofill <= ntofills)
            ci = tofill_ijs(1, itofill)
            cj = tofill_ijs(2, itofill)
            itofill = itofill + 1

            if (orders(ci, cj) == 0) then

                ! Check upstream cells to assign the current ones'
                ! order
                max_uorder = 0
                increase_order = .false.
                do iofs = 1, noffsets
                    ui = ci - offsets(iofs, 1)
                    uj = cj - offsets(iofs, 2)

                    ! Check bounds
                    if (array2d_oob(ui, uj, nrows, ncols)) cycle
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
            if (array2d_oob(ni, nj, nrows, ncols)) cycle
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

    subroutine flood_upstream( &
        flooded, dirs, seeds, valids, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        logical(kind=1), intent(in) :: seeds(nrows, ncols)
            !! Seed mask (true for seed cells)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the
            !! offsets
        ! Outputs
        logical(kind=1), intent(out) :: flooded(nrows, ncols)
            !! Mask indicating which cells are flooded (true for
            !! flooded cells, false for non-flooded cells)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Upstream-flooding queue capacity was exceeded
        ! Local variables
        integer :: iofs
            !! Index for iterating through offsets
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to noflow direction, used to
            !! identify seed cells
        integer :: iseed, nseeds, ifill, nfills
            !! Index for iterating through seed cells and buffer,
            !! and total number of seed cells and buffer fills
        integer :: si, sj, ci, cj, ui, uj
            !! Rows/columns for seed, current and upstream indices
        integer, allocatable :: seed_ids(:), tofill_ids(:)
            !! Buffers for storing linear IDs of seed cells and
            !! queued cells
        integer :: cell_id
        logical(kind=1) :: id_is_valid
        integer :: max_queue_size
            !! Maximum number of linear cell IDs in the seed and
            !! traversal queues
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
        call mask2id( &
            seeds, seed_ids, max_queue_size, nseeds, err_code)
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
            call id2ij_checked( &
                seed_ids(iseed), nrows, ncols, si, sj, id_is_valid)
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
                call id2ij_checked( &
                    tofill_ids(ifill), nrows, ncols, ci, cj, id_is_valid)
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
                    if (array2d_oob(ui, uj, nrows, ncols)) cycle
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
end module drainage_metrics
