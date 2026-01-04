subroutine label_flats_f( &
    z, labels, nrows, ncols, &
    seeds, nseeds, offsets, noffsets)
    implicit none
    integer, intent(in) :: nrows, ncols ! Size of the grid
    real, dimension(nrows, ncols), intent(in) :: z
    integer, dimension(nrows, ncols), intent(out) :: labels
    integer, intent(inout) :: nseeds
    integer, dimension(nseeds, 2), intent(inout) :: seeds ! Should have 1 added to match Fortran indexing
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets

    integer :: ilabel = 1
    integer, dimension(nrows*ncols, 2) :: tofill_buf
    integer :: ifill, nfills
    integer :: iseed = 1
    integer :: si, sj ! Seed indices
    integer :: ci, cj ! Current indices
    real :: sz ! Seed elevation
    integer :: iofs ! Offset index
    integer :: ni, nj ! Neighbour indices

    do while (iseed <= nseeds)
        si = seeds(iseed, 1)
        sj = seeds(iseed, 2)
        iseed = iseed + 1

        ! Skip if out of bounds
        if (si < 1 .or. si > nrows .or. sj < 1 .or. sj > ncols) then
            print *, "Warning: Skipping out of bound seed index (", si, ",", sj, ")"
            cycle
        end if
        ! Skip if already labeled
        if (labels(si, sj) /= 0) cycle

        sz = z(si, sj)

        ! Reset buffer
        ifill = 1
        nfills = 1
        tofill_buf(ifill, :) = [si, sj]
        labels(si, sj) = ilabel

        do while (ifill <= nfills)
            ci = tofill_buf(ifill, 1)
            cj = tofill_buf(ifill, 2)
            ifill = ifill + 1

            ! Loop over offsets to find connected flat cells
            do iofs = 1, noffsets
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                ! Check bounds
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                ! Check if already labeled
                if (labels(ni, nj) /= 0) cycle
                ! Check if same elevation
                if (z(ni, nj) /= sz) cycle
                ! Add to tofill buffer
                nfills = nfills + 1
                if (nfills > size(tofill_buf, 1)) then
                    print *, "Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(nfills, :) = [ni, nj]
                labels(ni, nj) = ilabel
            end do

        end do

        ilabel = ilabel + 1
    end do
end subroutine label_flats_f

subroutine away_from_high_loop_f( &
    z, labels, nrows, ncols, &
    high_edges, nedges, offsets, noffsets)
    implicit none
    integer, intent(in) :: nrows, ncols ! Size of the grid
    integer, dimension(nrows, ncols), intent(out) :: z
    integer, dimension(nrows, ncols), intent(in) :: labels ! Assume labels are 1 ... nlabels for flats, 0 for non-flats
    integer, intent(inout) :: nedges
    integer, dimension(nedges, 2), intent(inout) :: high_edges ! Should have 1 added to match Fortran indexing
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets

    integer :: nlabels ! number of unique labels
    integer, dimension(:), allocatable :: zmax ! max z per label
    integer, parameter, dimension(2) :: marker = [-1, -1]
    logical, dimension(nrows, ncols) :: queued
    logical :: added_since_marker = .false.
    integer :: nloops = 1
    integer :: iedge = 1 ! Index for high_edges
    integer :: jedge ! Index for high_edges
    integer :: iofs ! Offset index
    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbor indices

    ! Initialise high_edges buffer as a queue
    integer, dimension(count(labels /= 0) + max(nrows, ncols)*(maxval(labels) - minval(labels) + 1), 2) :: high_edges_buf
    high_edges_buf(1:nedges, :) = high_edges
    nedges = nedges + 1
    high_edges_buf(nedges, :) = marker

    nlabels = maxval(labels)
    allocate (zmax(nlabels))
    zmax = 0

    ! Initialise z to zero
    z = 0
    queued = .false.

    ! Mark initial seeds as queued
    do jedge = 1, nedges - 1
        ci = high_edges_buf(jedge, 1)
        cj = high_edges_buf(jedge, 2)
        queued(ci, cj) = .true.
    end do
    ! Loop through all high_edges to find cells flowing away from flats
    do while (iedge <= nedges)
        ci = high_edges_buf(iedge, 1)
        cj = high_edges_buf(iedge, 2)
        iedge = iedge + 1

        if (ci == marker(1) .and. cj == marker(2)) then
            ! Break if no more cells to process
            if (.not. added_since_marker) exit
            ! Skip if encountered marker
            nloops = nloops + 1
            nedges = nedges + 1
            ! Check buffer size
            if (nedges > size(high_edges_buf, 1)) then
                print *, "Error: High edges buffer overflow (size:", nedges, ", allocated:", size(high_edges_buf, 1), ")"
                stop
            end if
            high_edges_buf(nedges, :) = marker
            added_since_marker = .false.
            cycle
        end if

        ! Check bounds after marker check
        if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
            print *, "Error: Current indices out of bounds (", ci, ",", cj, ")"
            stop
        end if

        ! Queueing should guarantee we only visit each cell once
        z(ci, cj) = nloops
        if (labels(ci, cj) /= 0) then
            zmax(labels(ci, cj)) = nloops
        end if

        ! Loop over offsets to find contributing cells
        do iofs = 1, noffsets
            ! Skip self
            if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) cycle

            ni = ci + offsets(iofs, 1)
            nj = cj + offsets(iofs, 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Check if already queued
            if (queued(ni, nj)) cycle
            ! Check if is a flat
            if (labels(ni, nj) == 0) cycle
            ! Check if already processed
            if (z(ni, nj) > 0) cycle
            ! Check if neighbor is part of the same flat
            if (labels(ni, nj) /= labels(ci, cj)) cycle
            ! Update queue
            nedges = nedges + 1
            if (nedges > size(high_edges_buf, 1)) then
                print *, "Error: High edges buffer overflow (size:", nedges, ", allocated:", size(high_edges_buf, 1), ")"
                stop
            end if
            high_edges_buf(nedges, :) = [ni, nj]
            queued(ni, nj) = .true.
            added_since_marker = .true.
        end do
    end do
    ! Adjust z values within flats to ensure they flow away from high edges
    do concurrent(ci=1:nrows, cj=1:ncols, labels(ci, cj) /= 0)
        z(ci, cj) = zmax(labels(ci, cj)) - z(ci, cj) + 1
    end do
    ! do cj = 1, ncols
    !     do ci = 1, nrows
    !         if (labels(ci, cj) == 0) cycle
    !         z(ci, cj) = zmax(labels(ci, cj)) - z(ci, cj) + 1
    !     end do
    ! end do
end subroutine away_from_high_loop_f

subroutine towards_low_loop_f( &
    z, labels, nrows, ncols, &
    low_edges, nedges, offsets, noffsets)
    implicit none
    integer, intent(in) :: nrows, ncols ! Size of the grid
    integer, dimension(nrows, ncols), intent(out) :: z
    integer, dimension(nrows, ncols), intent(in) :: labels
    integer, intent(inout) :: nedges
    integer, dimension(nedges, 2), intent(inout) :: low_edges ! Should have 1 added to match Fortran indexing
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets

    integer, parameter, dimension(2) :: marker = [-1, -1]
    logical, dimension(nrows, ncols) :: queued
    logical :: added_since_marker = .false.
    integer :: nloops = 1
    integer :: iedge = 1 ! Index for low_edges
    integer :: jedge ! Index for low_edges
    integer :: iofs ! Offset index
    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbor indices

    ! Initialise low_edges buffer as a queue
    integer, dimension(count(labels /= 0) + max(nrows, ncols)*maxval(labels), 2) :: low_edges_buf
    low_edges_buf(1:nedges, :) = low_edges
    nedges = nedges + 1
    low_edges_buf(nedges, :) = marker

    ! Initialise z to zero
    z = 0
    queued = .false.

    ! Mark initial seeds as queued
    do jedge = 1, nedges - 1
        ci = low_edges_buf(jedge, 1)
        cj = low_edges_buf(jedge, 2)
        queued(ci, cj) = .true.
    end do

    ! Loop through all low_edges to find cells flowing into flats
    do while (iedge <= nedges)
        ci = low_edges_buf(iedge, 1)
        cj = low_edges_buf(iedge, 2)
        iedge = iedge + 1

        if (ci == marker(1) .and. cj == marker(2)) then
            ! Break if no more cells to process
            if (.not. added_since_marker) exit
            ! Skip if encountered marker
            nloops = nloops + 1
            nedges = nedges + 1
            ! Check buffer size
            if (nedges > size(low_edges_buf, 1)) then
                print *, "Error: Low edges buffer overflow (size:", nedges, ", allocated:", size(low_edges_buf, 1), ")"
                stop
            end if
            low_edges_buf(nedges, :) = marker
            added_since_marker = .false.
            cycle
        end if

        ! Check bounds after marker check
        if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
            print *, "Error: Current indices out of bounds (", ci, ",", cj, ")"
            stop
        end if

        ! Queueing should guarantee we only visit each cell once
        z(ci, cj) = nloops

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
            ! Check if is a flat
            if (labels(ni, nj) == 0) cycle
            ! Check if already processed
            if (z(ni, nj) > 0) cycle
            ! Check if neighbor is part of the same flat
            if (labels(ni, nj) /= labels(ci, cj)) cycle

            ! Update queue
            nedges = nedges + 1
            if (nedges > size(low_edges_buf, 1)) then
                print *, "Error: Low edges buffer overflow (size:", nedges, ", allocated:", size(low_edges_buf, 1), ")"
                stop
            end if
            low_edges_buf(nedges, :) = [ni, nj]
            queued(ni, nj) = .true.
            added_since_marker = .true.
        end do

    end do
end subroutine towards_low_loop_f
