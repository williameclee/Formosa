!!!
! Last modified
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Used function 'mask2id' as the linear-index version of
!       'mask2ij'
!   2026-08-05, En-Chi Lee (williameclee@gmail.com)
!     - Switched to 'iso_c_binding'
!!!

module drainage_basins
    use iso_c_binding, only: c_int8_t
    use utils, only: find_noflow_code, array2d_oob, mask2id, &
                     id2ij_checked, ij2id_checked
    implicit none(type, external)
contains
    subroutine label_watersheds( &
        labels, dirs, valids, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Assigns cells that drain into different sinks a unique
        !! label.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
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
        integer, intent(out) :: labels(nrows, ncols)
            !! Grid of watershed labels, where cells with the same
            !! label belong to the same watershed.
            !! Cells with no-data or that do not flow into any
            !! watershed should have a label of 0.
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Watershed traversal queue capacity was
            !!     exceeded
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
        logical(kind=1), allocatable :: seeds(:, :)
            !! Mask to identify seed cells for the algorithm (valid
            !! cells with noflow direction)
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
        labels = 0

        ! Append all cells with noflow direction to buffer
        max_queue_size = nrows*ncols
        allocate (seed_ids(max_queue_size), seeds(nrows, ncols), &
                  stat=err_code)
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
            labels(si, sj) = iseed
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

end module drainage_basins
