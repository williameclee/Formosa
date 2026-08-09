!!!
! Preparations before a digital elevation model (DEM) raster can be
! used to calculate flow direction and other geomorphological
! metrics.
!
! Content of this file is mostly designed to be called by the Python
! frontend and not directly by the user.
!
! Last modified: 2026-08-07, En-Chi Lee (williameclee@gmail.com)
!!!

module drainage_preprocessing
    use utils, only: array2d_oob, mask2id, &
                     id2ij_checked, ij2id_checked, &
                     push_priority_queue, pop_priority_queue
    implicit none(type, external)
    private :: fill_boundary_ocean_queue, fill_sink_priority_queue
contains
    pure subroutine fill_boundary_ocean_queue( &
        z, valids, nrows, ncols, seed_ids, nseeds, &
        ocean_lvl, flood_below, err_code)
        !! Pushes all boundary ocean cells of a DEM to the queue.
        !!
        !! Notes
        !! -----
        !! Private helper function for :func:'detect_ocean_basins_from_boundary'.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(nrows, ncols)
        real, intent(in) :: ocean_lvl
            !! Elevation of the ocean
        logical(kind=1), intent(in) :: flood_below
            !! Whether elevation below the 'ovean_lvl' should also
            !! be considered part of the ocean
        ! Outputs
        integer, intent(out) :: seed_ids(:)
        integer, intent(out) :: nseeds
        integer, intent(out) :: err_code
        ! Local variables
        integer :: si, sj

        ! Queue the boundary ocean cells
        err_code = 0
        nseeds = 0

        ! Leftmost column
        si = 1
        do sj = 1, ncols
            if (.not. valids(si, sj)) cycle
            ! Skip if not ocean
            if (z(si, sj) > ocean_lvl) cycle
            if ((.not. flood_below) .and. (z(si, sj) < ocean_lvl)) cycle
            if (nseeds >= size(seed_ids, dim=1)) then
                err_code = 3
                return
            end if
            nseeds = nseeds + 1
            seed_ids(nseeds) = ij2id_checked(si, sj, nrows, ncols)
        end do
        ! Rightmost column
        si = nrows
        do sj = 1, ncols
            if (.not. valids(si, sj)) cycle
            if (z(si, sj) > ocean_lvl) cycle
            if ((.not. flood_below) .and. (z(si, sj) < ocean_lvl)) cycle
            if (nseeds >= size(seed_ids, dim=1)) then
                err_code = 3
                return
            end if
            nseeds = nseeds + 1
            seed_ids(nseeds) = ij2id_checked(si, sj, nrows, ncols)
        end do
        ! Top row
        sj = 1
        do si = 2, nrows - 1
            if (.not. valids(si, sj)) cycle
            if (z(si, sj) > ocean_lvl) cycle
            if ((.not. flood_below) .and. (z(si, sj) < ocean_lvl)) cycle
            if (nseeds >= size(seed_ids, dim=1)) then
                err_code = 3
                return
            end if
            nseeds = nseeds + 1
            seed_ids(nseeds) = ij2id_checked(si, sj, nrows, ncols)
        end do
        ! Bottom row
        sj = ncols
        do si = 2, nrows - 1
            if (.not. valids(si, sj)) cycle
            if (z(si, sj) > ocean_lvl) cycle
            if ((.not. flood_below) .and. (z(si, sj) < ocean_lvl)) cycle
            if (nseeds >= size(seed_ids, dim=1)) then
                err_code = 3
                return
            end if
            nseeds = nseeds + 1
            seed_ids(nseeds) = ij2id_checked(si, sj, nrows, ncols)
        end do
    end subroutine fill_boundary_ocean_queue

    pure subroutine detect_ocean_basins_from_boundary( &
        z, valids, basins, nrows, ncols, offsets, noffsets, &
        ocean_lvl, flood_below, err_code)
        !! Finds ocean basins the border the DEM's edges, and gives
        !! each a unique label.
        !!
        !! An ocean basin is identified by elevation at or at or
        !! below a given threshold.
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
        real, intent(in) :: ocean_lvl
            !! Elevation of the ocean
        logical(kind=1), intent(in) :: flood_below
            !! Whether elevation below the 'ovean_lvl' should also
            !! be considered part of the ocean
        ! Outputs
        integer, intent(out) :: basins(nrows, ncols)
            !! Basin ID grid
        integer, intent(out) :: err_code
        ! Local variables
        logical(kind=1), allocatable :: processed(:, :)
        integer :: ibasin
        integer, allocatable :: seed_ids(:), basin_ids(:)
        integer :: si, sj, sid, ci, cj, cid, ni, nj, nid
        integer :: iseed, nseeds, icell, ncells
        integer :: iofs
        logical(kind=1) :: is_valid

        allocate (seed_ids((nrows + ncols)*2 - 2), &
                  basin_ids(nrows*ncols), processed(nrows, ncols), &
                  stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            if (allocated(seed_ids)) deallocate (seed_ids)
            if (allocated(processed)) deallocate (processed)
            return
        end if

        ! Queue the boundary ocean cells
        call fill_boundary_ocean_queue( &
            z, valids, nrows, ncols, seed_ids, nseeds, &
            ocean_lvl, flood_below, err_code)
        if (err_code /= 0) return

        basins = 0
        if (nseeds == 0) return

        ! Flood through the seeds
        processed = .false.
        ibasin = 0
        iseed = 1
        do while (iseed <= nseeds)
            sid = seed_ids(iseed)
            call id2ij_checked(sid, nrows, ncols, si, sj, is_valid)
            if (.not. is_valid) then
                iseed = iseed + 1
                cycle
            elseif (processed(si, sj)) then
                iseed = iseed + 1
                cycle
            end if

            ! A new basin is found
            ibasin = ibasin + 1
            basins(si, sj) = ibasin
            processed(si, sj) = .true.

            ! Start flooding
            icell = 1
            ncells = 1
            cid = sid
            basin_ids(icell) = cid

            do while (icell <= ncells)
                cid = basin_ids(icell)
                call id2ij_checked(cid, nrows, ncols, ci, cj, is_valid)

                ! Loop through neighbours
                do iofs = 1, noffsets
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    if (.not. valids(ni, nj)) cycle
                    if (processed(ni, nj)) cycle
                    ! Check if is ocean
                    if (z(ni, nj) > ocean_lvl) cycle
                    if ((.not. flood_below) .and. (z(ni, nj) < ocean_lvl)) cycle
                    ncells = ncells + 1
                    nid = ij2id_checked(ni, nj, nrows, ncols)
                    basin_ids(ncells) = nid
                    processed(ni, nj) = .true.
                    basins(ni, nj) = ibasin
                end do

                icell = icell + 1
            end do

            iseed = iseed + 1
        end do
    end subroutine detect_ocean_basins_from_boundary

    pure subroutine fill_sink_priority_queue( &
        z, valids, more_sinks, processed, pqueue, pqueue_size, &
        offsets, err_code)
        !! Pushes sink cells of a DEM to the priority queue.
        !!
        !! The following kinds of cells are considered sinks:
        !!  1. Valid edge cells of the DEM
        !!  2. Valid cells surrounding an invalid cell
        !!  3. Additional valid sink cells specified as 'more_sinks'
        !!
        !! Notes
        !! -----
        !! This is a private helper function for :func:'fill_depressions'.
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(:, :)
            !! Validity mask (false for no-data)
        logical(kind=1), intent(in) :: more_sinks(:, :)
            !! Additional sink cell mask
        logical(kind=1), intent(inout) :: processed(:, :)
        integer, intent(inout) :: pqueue(:)
        integer, intent(inout) :: pqueue_size
        integer, intent(in) :: offsets(:, :)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: err_code
        ! Local variables
        integer :: nrows, ncols, noffsets
            !! Size of the grid
        integer :: ci, cj, cid, ni, nj, nid
        integer :: iofs

        nrows = size(z, dim=1)
        ncols = size(z, dim=2)
        noffsets = size(offsets, dim=1)

        ! Push all valid boundaries to the queue
        ! Top row
        ci = 1
        do cj = 1, ncols
            if (.not. valids(ci, cj)) cycle
            if (processed(ci, cj)) cycle
            cid = ij2id_checked(ci, cj, nrows, ncols)
            call push_priority_queue( &
                pqueue, pqueue_size, cid, z, err_code)
            if (err_code /= 0) return
            processed(ci, cj) = .true.
        end do
        ! Bottom row
        ci = nrows
        do cj = 1, ncols
            if (.not. valids(ci, cj)) cycle
            if (processed(ci, cj)) cycle
            cid = ij2id_checked(ci, cj, nrows, ncols)
            call push_priority_queue( &
                pqueue, pqueue_size, cid, z, err_code)
            if (err_code /= 0) return
            processed(ci, cj) = .true.
        end do
        ! Leftmost column
        cj = 1
        do ci = 1, nrows
            if (.not. valids(ci, cj)) cycle
            if (processed(ci, cj)) cycle
            cid = ij2id_checked(ci, cj, nrows, ncols)
            call push_priority_queue( &
                pqueue, pqueue_size, cid, z, err_code)
            if (err_code /= 0) return
            processed(ci, cj) = .true.
        end do
        ! Rightmost column
        cj = ncols
        do ci = 1, nrows
            if (.not. valids(ci, cj)) cycle
            if (processed(ci, cj)) cycle
            cid = ij2id_checked(ci, cj, nrows, ncols)
            call push_priority_queue( &
                pqueue, pqueue_size, cid, z, err_code)
            if (err_code /= 0) return
            processed(ci, cj) = .true.
        end do

        ! Queue neighbours of non-valid cells (which may be ocean,
        ! etc.) or additional sinks
        do cj = 1, ncols
            do ci = 1, nrows
                if (valids(ci, cj)) then
                    if (processed(ci, cj)) cycle
                    if (.not. more_sinks(ci, cj)) cycle
                    ! Queue the sink
                    call push_priority_queue( &
                        pqueue, pqueue_size, &
                        ij2id_checked(ci, cj, nrows, ncols), z, err_code)
                    if (err_code /= 0) return
                    processed(ci, cj) = .true.
                    cycle
                end if
                ! Push all neighbours to the queue
                do iofs = 1, noffsets
                    ! In opposite direction since we want to find
                    ! cells that can flow to these invalid cells
                    ni = ci - offsets(iofs, 1)
                    nj = cj - offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    ! Skip if not valid or already processed
                    if (.not. valids(ni, nj)) cycle
                    if (processed(ni, nj)) cycle

                    ! Push to the queue
                    nid = ij2id_checked(ni, nj, nrows, ncols)
                    call push_priority_queue( &
                        pqueue, pqueue_size, nid, z, err_code)
                    if (err_code /= 0) return
                    processed(ni, nj) = .true.
                end do
            end do
        end do
    end subroutine fill_sink_priority_queue

    pure subroutine fill_depressions( &
        z, valids, more_sinks, z_filled, nrows, ncols, &
        offsets, noffsets, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        real, intent(in) :: z(nrows, ncols)
            !! Elevation grid
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        logical(kind=1), intent(in) :: more_sinks(nrows, ncols)
            !! Additional sink mask
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        ! Outputs
        real, intent(out) :: z_filled(nrows, ncols)
            !! Depression-filled elevation grid
        integer, intent(out) :: err_code
        ! Local variables
        integer :: ci, cj, cid, ni, nj, nid
        integer :: iofs
        logical(kind=1) :: is_valid
        logical(kind=1), allocatable :: processed(:, :)
        integer, allocatable :: pqueue(:)
        integer :: pqueue_size

        allocate (processed(nrows, ncols), pqueue(nrows*ncols), &
                  stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            if (allocated(processed)) deallocate (processed)
            if (allocated(pqueue)) deallocate (pqueue)
            return
        end if

        z_filled = z
        processed = .false.
        pqueue_size = 0

        ! Push all valid boundaries to the queue
        call fill_sink_priority_queue( &
            z, valids, more_sinks, processed, &
            pqueue, pqueue_size, offsets, err_code)
        if (err_code /= 0) return

        ! Start processing the cells
        do while (pqueue_size > 0)
            call pop_priority_queue( &
                pqueue, pqueue_size, cid, z_filled, err_code)
            if (err_code /= 0) return
            call id2ij_checked(cid, nrows, ncols, ci, cj, is_valid)
            do iofs = 1, noffsets
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                ! Check bounds
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                ! Skip if not valid or already processed
                if (.not. valids(ni, nj)) cycle
                if (processed(ni, nj)) cycle
                z_filled(ni, nj) = max(z(ni, nj), z_filled(ci, cj))
                processed(ni, nj) = .true.
                nid = ij2id_checked(ni, nj, nrows, ncols)
                call push_priority_queue( &
                    pqueue, pqueue_size, nid, z_filled, err_code)
                if (err_code /= 0) return
            end do
        end do
    end subroutine fill_depressions

    pure subroutine label_mask_areas( &
        mask, labels, nrows, ncols, offsets, nofss, err_code)
        !! Finds connected mask areas and assigns each a unique
        !! label.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        logical(kind=1), intent(in) :: mask(nrows, ncols)
        integer, intent(in) :: nofss
            !! Number of flow directions
        integer, intent(in) :: offsets(nofss, 2)
            !! List of offsets for each flow direction
        ! Outputs
        integer, intent(out) :: labels(nrows, ncols)
        integer, intent(out) :: err_code
        ! Local variables
        integer :: queue_size
        integer, allocatable :: seed_queue(:), flood_queue(:)
        integer :: iseed, nseeds, icell, ncells
        integer :: si, sj, ci, cj, ni, nj
        integer :: iofs
        integer :: ilabel
        logical(kind=1) :: is_valid

        queue_size = count(mask)
        allocate (seed_queue(queue_size), flood_queue(queue_size), &
                  stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if

        labels = 0
        if (queue_size == 0) return

        ! Fill the seed queue
        call mask2id(mask, seed_queue, queue_size, nseeds, err_code)
        if (err_code /= 0) return

        ! Go through each cell
        ilabel = 0
        iseed = 1
        do while (iseed <= nseeds)
            call id2ij_checked( &
                seed_queue(iseed), nrows, ncols, si, sj, is_valid)
            if (labels(si, sj) /= 0) then
                iseed = iseed + 1
                cycle
            end if
            ! A new area is found
            ilabel = ilabel + 1
            labels(si, sj) = ilabel
            ncells = 1
            icell = 1
            flood_queue(icell) = seed_queue(iseed)

            do while (icell <= ncells)
                call id2ij_checked( &
                    flood_queue(icell), nrows, ncols, ci, cj, is_valid)

                ! Loop through neighbours
                do iofs = 1, nofss
                    ni = ci + offsets(iofs, 1)
                    nj = cj + offsets(iofs, 2)
                    ! Check bounds
                    if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    if (.not. mask(ni, nj)) cycle
                    if (labels(ni, nj) /= 0) cycle
                    ncells = ncells + 1
                    flood_queue(ncells) = ij2id_checked(ni, nj, nrows, ncols)
                    labels(ni, nj) = ilabel
                end do
                icell = icell + 1
            end do
            iseed = iseed + 1
        end do
    end subroutine label_mask_areas
end module drainage_preprocessing
