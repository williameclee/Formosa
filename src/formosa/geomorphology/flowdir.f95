module flow_utils
    implicit none
contains
    subroutine find_noflow_code( &
        offsets, codes, noffsets, noflow_code)
        implicit none
        integer, intent(in) :: noffsets
        integer, dimension(noffsets, 2), intent(in) :: offsets
        integer, dimension(noffsets), intent(in) :: codes
        integer, intent(out) :: noflow_code

        integer :: iofs ! Offset index
        do iofs = 1, noffsets
            if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) then
                noflow_code = codes(iofs)
                exit
            end if
        end do
    end subroutine find_noflow_code
    subroutine make_offset_lookups(offsets, codes, noffsets, mincode, maxcode, diffs)
        implicit none
        ! Inputs
        integer, intent(in) :: noffsets
        integer, dimension(noffsets, 2), intent(in) :: offsets
        integer, dimension(noffsets), intent(in) :: codes
        integer, intent(in) :: mincode, maxcode
        ! Outputs
        integer, dimension(mincode:maxcode, 2), intent(out) :: diffs ! Lookup table for offsets

        integer :: iofs

        ! Create lookup tables for offsets
        do iofs = 1, noffsets
            diffs(codes(iofs), 1) = offsets(iofs, 1)
            diffs(codes(iofs), 2) = offsets(iofs, 2)
        end do
    end subroutine make_offset_lookups
    subroutine mask2ij( &
        mask, nrows, ncols, ij, nij, cnt)
        ! TODO: Optimise this subroutine?
        implicit none
        integer, intent(in) :: nrows, ncols
        logical, dimension(nrows, ncols), intent(in) :: mask
        integer, intent(in) :: nij
        integer, dimension(nij, 2), intent(out) :: ij
        integer, intent(out) :: cnt

        integer :: ci, cj

        ! Count number of valid neighbors
        cnt = 0
        do cj = 1, ncols
            do ci = 1, nrows
                if (mask(ci, cj)) then
                    cnt = cnt + 1
                    if (cnt <= nij) then
                        ij(cnt, 1) = ci
                        ij(cnt, 2) = cj
                    end if
                end if
            end do
        end do
    end subroutine mask2ij
end module flow_utils

subroutine compute_flowdir_simple_f( &
    z, valids, flowdir, is_flat, nrows, ncols, &
    offsets, codes, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols ! Size of the grid
    real, dimension(nrows, ncols), intent(in) :: z
    logical, dimension(nrows, ncols), intent(in) :: valids
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes
    ! Outputs
    integer, dimension(nrows, ncols), intent(out) :: flowdir
    logical, dimension(nrows, ncols), intent(out) :: is_flat

    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbour indices
    integer :: iofs ! Offset index
    real :: zmin
    integer :: noflow_code = 0 ! Assume 0 is noflow unless found otherwise

    ! Find noflow code
    call find_noflow_code(offsets, codes, noffsets, noflow_code)

    !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, iofs, ni, nj, zmin) &
    !$omp COLLAPSE(2) &
    !$omp SCHEDULE(STATIC)
    do ci = 1, nrows
        do cj = 1, ncols
            if (.not. valids(ci, cj)) then
                flowdir(ci, cj) = noflow_code
                cycle
            end if

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
                    flowdir(ci, cj) = codes(iofs)
                end if
            end do
        end do
    end do
    !$omp END PARALLEL DO

    ! Identify flat cells
    !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj) &
    !$omp COLLAPSE(2) &
    !$omp SCHEDULE(STATIC)
    do ci = 1, nrows
        do cj = 1, ncols
            if (.not. valids(ci, cj)) then
                is_flat(ci, cj) = .false.
            else if (flowdir(ci, cj) == noflow_code) then
                is_flat(ci, cj) = .true.
            else
                is_flat(ci, cj) = .false.
            end if
        end do
    end do
end subroutine compute_flowdir_simple_f

subroutine compute_masked_flowdir_f( &
    z, labels, flowdir, nrows, ncols, &
    offsets, codes, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols ! Size of the grid
    integer, dimension(nrows, ncols), intent(in) :: z, labels
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes
    ! Outputs
    integer, dimension(nrows, ncols), intent(out) :: flowdir

    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbour indices
    integer :: iofs ! Offset index
    integer :: zmin
    integer :: noflow_code = 0 ! Assume 0 is noflow unless found otherwise

    ! Find noflow code
    call find_noflow_code(offsets, codes, noffsets, noflow_code)

    !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, iofs, ni, nj, zmin) &
    !$omp COLLAPSE(2) &
    !$omp SCHEDULE(STATIC)
    do ci = 1, nrows
        do cj = 1, ncols
            if (labels(ci, cj) == 0) then
                flowdir(ci, cj) = noflow_code
                cycle
            end if

            zmin = z(ci, cj)

            do iofs = 1, noffsets
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                ! Check bounds
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                ! Check if neighbour is part of the same flat
                if (labels(ni, nj) /= labels(ci, cj)) cycle
                ! Check if neighbour has lower elevation
                if (z(ni, nj) < zmin) then
                    zmin = z(ni, nj)
                    flowdir(ci, cj) = codes(iofs)
                end if
            end do
        end do
    end do
    !$omp END PARALLEL DO
end subroutine compute_masked_flowdir_f

subroutine find_flat_edges_f( &
    z, flowdir, valids, is_low_edge, is_high_edge, nrows, ncols, &
    offsets, codes, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols ! Size of the grid
    real, dimension(nrows, ncols), intent(in) :: z
    integer, dimension(nrows, ncols), intent(in) :: flowdir
    logical, dimension(nrows, ncols), intent(in) :: valids
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes
    ! Outputs
    logical, dimension(nrows, ncols), intent(out) :: is_low_edge, is_high_edge

    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbour indices
    integer :: iofs ! Offset index
    integer :: noflow_code = 0 ! Assume 0 is noflow unless found otherwise

    ! Find noflow code
    call find_noflow_code(offsets, codes, noffsets, noflow_code)

    !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, iofs, ni, nj) &
    !$omp COLLAPSE(2) &
    !$omp SCHEDULE(STATIC)
    do ci = 1, nrows
        do cj = 1, ncols
            if (.not. valids(ci, cj)) then
                is_high_edge(ci, cj) = .false.
                is_low_edge(ci, cj) = .false.
                cycle
            end if

            do iofs = 1, noffsets
                ni = ci + offsets(iofs, 1)
                nj = cj + offsets(iofs, 2)
                ! Check bounds
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                ! Check if neighbour is part of the same flat
                if (.not. valids(ni, nj)) cycle
                ! Check for low edge
                if (flowdir(ci, cj) /= noflow_code .and. flowdir(ni, nj) == noflow_code .and. z(ci, cj) == z(ni, nj)) then
                    is_low_edge(ci, cj) = .true.
                    is_high_edge(ci, cj) = .false.
                    exit
                end if
                ! Check for high edge
                if (flowdir(ci, cj) == noflow_code .and. z(ci, cj) < z(ni, nj)) then
                    is_high_edge(ci, cj) = .true.
                    is_low_edge(ci, cj) = .false.
                    exit
                end if
            end do
            ! If neither edge type found
            if (.not. is_high_edge(ci, cj) .and. .not. is_low_edge(ci, cj)) then
                is_high_edge(ci, cj) = .false.
                is_low_edge(ci, cj) = .false.
            end if
        end do
    end do
    !$omp END PARALLEL DO
end subroutine find_flat_edges_f

subroutine label_flats_f( &
    z, is_seed, labels, nrows, ncols, &
    offsets, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols ! Size of the grid
    real, dimension(nrows, ncols), intent(in) :: z
    logical, dimension(nrows, ncols), intent(in) :: is_seed
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    ! Outputs
    integer, dimension(nrows, ncols), intent(out) :: labels

    integer :: ilabel = 1
    integer, dimension(:, :), allocatable :: tofill_buf
    integer :: ifill, nfills
    integer :: iseed = 1
    integer :: si, sj ! Seed indices
    integer :: ci, cj ! Current indices
    real :: sz ! Seed elevation
    integer :: iofs ! Offset index
    integer :: ni, nj ! Neighbour indices

    ! Convert is_seed mask to list of seed indices
    integer, dimension(:, :), allocatable :: seeds
    integer :: nseeds
    allocate (tofill_buf(nrows*ncols, 2))
    allocate (seeds(nrows*ncols, 2))
    call mask2ij( &
        is_seed, nrows, ncols, &
        seeds, size(seeds, dim=1), nseeds)

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
    deallocate (tofill_buf)
    deallocate (seeds)
end subroutine label_flats_f

subroutine away_from_high_loop_f( &
    z, labels, nrows, ncols, &
    is_high_edge, offsets, noffsets)
    use flow_utils
    implicit none
    integer, intent(in) :: nrows, ncols ! Size of the grid
    integer, dimension(nrows, ncols), intent(out) :: z
    integer, dimension(nrows, ncols), intent(in) :: labels ! Assume labels are 1 ... nlabels for flats, 0 for non-flats
    logical, dimension(nrows, ncols), intent(in) :: is_high_edge
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
    integer :: nedges
    integer, dimension(:, :), allocatable :: high_edges_buf
    allocate (high_edges_buf(count(labels /= 0) + max(nrows, ncols)*(maxval(labels) - minval(labels) + 1), 2))

    call mask2ij( &
        is_high_edge, nrows, ncols, &
        high_edges_buf, size(high_edges_buf, dim=1), nedges)
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
end subroutine away_from_high_loop_f

subroutine towards_low_loop_f( &
    z, labels, nrows, ncols, &
    is_low_edge, offsets, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols ! Size of the grid
    integer, dimension(nrows, ncols), intent(in) :: labels
    logical, dimension(nrows, ncols), intent(in) :: is_low_edge
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    ! Outputs
    integer, dimension(nrows, ncols), intent(out) :: z

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
    integer :: nedges
    integer, dimension(:, :), allocatable :: low_edges_buf
    allocate (low_edges_buf(count(labels /= 0) + max(nrows, ncols)*maxval(labels), 2))

    call mask2ij( &
        is_low_edge, nrows, ncols, &
        low_edges_buf, size(low_edges_buf, dim=1), nedges)
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

subroutine compute_indegree_f( &
    flowdir, indegree, nrows, ncols, &
    offsets, codes, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols
    integer, dimension(nrows, ncols), intent(in) :: flowdir
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes
    ! Outputs
    integer, dimension(nrows, ncols), intent(out) :: indegree

    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbour indices
    integer, dimension(:, :), allocatable :: diffs ! Lookup tables for offsets
    integer :: code, mincode, maxcode

    ! Create lookup tables for offsets
    mincode = minval(codes)
    maxcode = maxval(codes)
    allocate (diffs(mincode:maxcode, 2))
    call make_offset_lookups(offsets, codes, noffsets, mincode, maxcode, diffs)

    indegree = 0

    !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj) &
    !$omp COLLAPSE(2) &
    !$omp SCHEDULE(STATIC)
    do ci = 1, nrows
        do cj = 1, ncols
            ! Get neighbour indices based on flow direction
            code = flowdir(ci, cj)
            if (code < mincode .or. code > maxcode) cycle
            ni = ci + diffs(code, 1)
            nj = cj + diffs(code, 2)

            ! Check bounds
            if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
            ! Skip self-loops
            if (ni == ci .and. nj == cj) cycle

            ! Increment indegree of downstream cell, make sure only one thread updates at a time
            !$omp ATOMIC UPDATE
            indegree(ni, nj) = indegree(ni, nj) + 1
            !$omp END ATOMIC
        end do
    end do
    !$omp END PARALLEL DO
    deallocate (diffs)
end subroutine compute_indegree_f

subroutine compute_accumulation_f( &
    flowdir, valids, weights, indegrees, accumulations, nrows, ncols, &
    offsets, codes, noffsets)
    use flow_utils
    implicit none
    ! Inputs
    integer, intent(in) :: nrows, ncols
    integer, dimension(nrows, ncols), intent(in) :: flowdir
    logical, dimension(nrows, ncols), intent(in) :: valids
    real, dimension(nrows, ncols), intent(in) :: weights
    integer, dimension(nrows, ncols), intent(inout) :: indegrees
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes
    ! Outputs
    real, dimension(nrows, ncols), intent(out) :: accumulations

    logical, dimension(:, :), allocatable :: is_tofill_seed
    integer, dimension(:, :), allocatable :: tofill_buf
    integer :: itofill, ntofills
    integer :: ci, cj ! Current indices
    integer :: ni, nj ! Neighbour indices
    integer, dimension(:, :), allocatable :: diffs ! Lookup tables for offsets
    integer :: code, mincode, maxcode

    ! Create lookup tables for offsets
    mincode = minval(codes)
    maxcode = maxval(codes)
    allocate (diffs(mincode:maxcode, 2))
    call make_offset_lookups(offsets, codes, noffsets, mincode, maxcode, diffs)

    ! Fill the tofill buffer with all valid cells with zero indegree
    allocate (tofill_buf(nrows*ncols, 2))
    allocate (is_tofill_seed(nrows, ncols))
    is_tofill_seed = valids .and. (indegrees == 0)
    call mask2ij(is_tofill_seed, &
                 nrows, ncols, &
                 tofill_buf, nrows*ncols, ntofills)
    deallocate (is_tofill_seed)

    accumulations = weights
    itofill = 1
    do while (itofill <= ntofills)
        ci = tofill_buf(itofill, 1)
        cj = tofill_buf(itofill, 2)
        itofill = itofill + 1

        code = flowdir(ci, cj)
        if (code < mincode .or. code > maxcode) cycle
        ni = ci + diffs(flowdir(ci, cj), 1)
        nj = cj + diffs(flowdir(ci, cj), 2)

        ! Check bounds
        if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
        ! Check mask
        if (.not. valids(ni, nj)) cycle
        ! Check not a self-loop
        if (ni == ci .and. nj == cj) cycle
        ! Check not already processed
        if (indegrees(ni, nj) <= 0) cycle

        ! Update accumulation of downstream cell
        accumulations(ni, nj) = accumulations(ni, nj) + accumulations(ci, cj)
        ! Decrement indegree of downstream cell
        indegrees(ni, nj) = indegrees(ni, nj) - 1
        ! If indegree is zero, add to tofill buffer
        if (indegrees(ni, nj) == 0) then
            ntofills = ntofills + 1
            if (ntofills > size(tofill_buf, 1)) then
                print *, "Error: tofill buffer overflow (size:", ntofills, ", allocated:", size(tofill_buf, 1), ")"
                stop
            end if
            tofill_buf(ntofills, :) = [ni, nj]
        end if
    end do
    deallocate (tofill_buf)
    deallocate (diffs)
end subroutine compute_accumulation_f

subroutine compute_back_distance_f( &
    dist, flowdir, x, y, valid, nrows, ncols, offsets, codes, noffsets)
    use omp_lib
    use flow_utils
    implicit none
    integer, intent(in) :: nrows, ncols
    integer, dimension(nrows, ncols), intent(in) :: flowdir
    real, dimension(nrows, ncols), intent(in) :: x, y
    logical, dimension(nrows, ncols), intent(in) :: valid
    real, dimension(nrows, ncols), intent(out) :: dist
    integer, intent(in) :: noffsets
    integer, dimension(noffsets, 2), intent(in) :: offsets
    integer, dimension(noffsets), intent(in) :: codes

    integer, dimension(:, :), allocatable :: seed_buf
    integer :: iseed, nseeds
    integer, dimension(:, :), allocatable :: tofill_buf
    logical, dimension(:, :), allocatable :: is_seed
    integer :: ifill, nfills
    integer :: si, sj ! Seed indices
    integer :: ci, cj ! Current indices
    integer :: ui, uj ! Upstream indices
    integer :: iofs ! Offset index
    integer :: noflow_code = 0 ! Assume 0 is noflow unless found otherwise

    ! Find noflow code
    call find_noflow_code(offsets, codes, noffsets, noflow_code)

    dist = -1

    ! Append all cells with noflow direction to buffer
    allocate (seed_buf(nrows*ncols, 2))
    allocate (is_seed(nrows, ncols))
    is_seed = valid .and. (flowdir == noflow_code)
    call mask2ij(is_seed, nrows, ncols, &
                 seed_buf, nrows*ncols, nseeds)
    deallocate (is_seed)

    ! Loop through seeds
    !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_buf)
    allocate (tofill_buf(nrows*ncols, 2))
    !$omp DO SCHEDULE(DYNAMIC)
    do iseed = 1, nseeds
        si = seed_buf(iseed, 1)
        sj = seed_buf(iseed, 2)

        ! Loop through buffer
        nfills = 1
        ifill = 1
        dist(si, sj) = 0.0
        tofill_buf(1, :) = [si, sj]

        do while (ifill <= nfills)
            ci = tofill_buf(ifill, 1)
            cj = tofill_buf(ifill, 2)
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
                if (.not. valid(ui, uj)) cycle
                ! Check if already assigned
                if (dist(ui, uj) >= 0) cycle
                ! Check if flows into current cell
                if (flowdir(ui, uj) /= codes(iofs)) cycle

                ! Add to buffer
                nfills = nfills + 1
                if (nfills > size(tofill_buf, 1)) then
                    print *, "Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(nfills, :) = [ui, uj]
                ! Compute distance
                dist(ui, uj) = dist(ci, cj) + sqrt( &
                               (x(ui, uj) - x(ci, cj))**2 + &
                               (y(ui, uj) - y(ci, cj))**2)
            end do
        end do
    end do
    !$omp END DO
    deallocate (tofill_buf)
    !$omp END PARALLEL
end subroutine compute_back_distance_f
