module flowdir_utils
    implicit none
contains
    function find_noflow_code(offsets, codes, noffsets, default_noflow_code) result(noflow_code)
        implicit none
        integer, intent(in) :: noffsets
        integer, intent(in) :: offsets(noffsets, 2)
        integer*1, intent(in) :: codes(noffsets)
        integer*1, intent(in), optional :: default_noflow_code
        integer*1 :: noflow_code

        integer :: iofs ! Offset index

        if (present(default_noflow_code)) then
            noflow_code = default_noflow_code
        else
            noflow_code = 0
        end if

        do iofs = 1, noffsets
            if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) then
                noflow_code = codes(iofs)
                exit
            end if
        end do
    end function find_noflow_code

    function find_opposite_codes(offsets, codes, noffsets) result(opp_codes)
        implicit none
        integer, intent(in) :: noffsets
        integer, intent(in) :: offsets(noffsets, 2)
        integer*1, intent(in) :: codes(noffsets)
        integer*1 :: opp_codes(noffsets)
        integer :: iofs, jofs

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
        implicit none
        ! Inputs
        integer, intent(in) :: noffsets
        integer, intent(in) :: offsets(noffsets, 2)
        integer*1, intent(in) :: codes(noffsets)
        ! Outputs
        integer :: diffs(0:255, 2) ! Lookup table for offsets

        integer :: iofs

        ! Create lookup tables for offsets
        diffs = -99 ! Initialise to invalid value
        do iofs = 1, noffsets
            if (codes(iofs) < 0 .or. codes(iofs) > 255) then
                print *, "[OFFSET_LOOKUP] Error: Flow direction code out of bounds: ", codes(iofs)
                stop
            end if
            diffs(codes(iofs), 1) = offsets(iofs, 1)
            diffs(codes(iofs), 2) = offsets(iofs, 2)
        end do
    end function fill_offset_lookup

    subroutine mask2ij( &
        mask, nrows, ncols, ij, nij, cnt)
        ! TODO: Optimise this subroutine?
        implicit none
        integer, intent(in) :: nrows, ncols
        logical*1, dimension(nrows, ncols), intent(in) :: mask
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
end module flowdir_utils

module flowdir
    use omp_lib
    use flowdir_utils
    implicit none
contains
    subroutine compute_flowdir_simple( &
        z, valids, flowdir, is_flat, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        real, intent(in) :: z(nrows, ncols)
        logical*1, intent(in) :: valids(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        integer*1, intent(in) :: codes(noffsets)
        ! Outputs
        integer*1, intent(out) :: flowdir(nrows, ncols)
        logical*1, intent(out) :: is_flat(nrows, ncols)

        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer :: iofs ! Offset index
        real :: zmin
        integer*1 :: noflow_code ! Assume 0 is noflow unless found otherwise

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

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
    end subroutine compute_flowdir_simple

    subroutine compute_masked_flowdir( &
        z, labels, flowdir, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer, intent(in) :: z(nrows, ncols), labels(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        integer*1, intent(in) :: codes(noffsets)
        ! Outputs
        integer*1, intent(out) :: flowdir(nrows, ncols)

        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer :: iofs ! Offset index
        integer :: zmin
        integer*1 :: noflow_code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

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
    end subroutine compute_masked_flowdir

    subroutine find_flat_edges( &
        z, flowdir, valids, is_low_edge, is_high_edge, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        real, intent(in) :: z(nrows, ncols)
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valids(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        logical*1, intent(out) :: is_low_edge(nrows, ncols), is_high_edge(nrows, ncols)

        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer :: iofs ! Offset index
        integer*1 :: noflow_code

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

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
    end subroutine find_flat_edges

    subroutine label_flats( &
        z, is_seed, labels, nrows, ncols, &
        offsets, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        real, intent(in) :: z(nrows, ncols)
        logical*1, intent(in) :: is_seed(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer, intent(out) :: labels(nrows, ncols)

        integer :: label, iseed, nseeds, ifill, nfills
        integer :: si, sj, ci, cj, ni, nj ! Seed, current, neighbour indices
        real :: sz ! Seed elevation
        integer :: iofs ! Offset index
        integer, allocatable :: tofill_buf(:, :), seeds(:, :)

        allocate (tofill_buf(nrows*ncols, 2))
        allocate (seeds(nrows*ncols, 2))
        call mask2ij( &
            is_seed, nrows, ncols, &
            seeds, size(seeds, dim=1), nseeds)

        label = 1
        iseed = 1
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
            labels(si, sj) = label

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
                      print *, "[LABEL_FLAT] Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                        stop
                    end if
                    tofill_buf(nfills, :) = [ni, nj]
                    labels(ni, nj) = label
                end do

            end do

            label = label + 1
        end do
        deallocate (tofill_buf)
        deallocate (seeds)
    end subroutine label_flats

    subroutine away_from_high( &
        z, labels, nrows, ncols, &
        is_high_edge, offsets, noffsets)
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer, intent(in) :: labels(nrows, ncols)
        logical*1, intent(in) :: is_high_edge(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer, intent(out) :: z(nrows, ncols)

        integer :: nlabels ! number of unique labels
        integer :: nloops
        integer :: iedge, nedges ! Index for high_edges
        integer :: iofs ! Offset index
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        logical*1 :: added_since_marker
        integer, parameter :: marker(2) = [-1, -1]
        logical*1, allocatable :: queued(:, :)
        integer, allocatable :: zmax(:) ! max z per label
        integer, allocatable :: high_edges_buf(:, :) ! Queue buffer

        allocate (queued(nrows, ncols))
        allocate (high_edges_buf(count(labels /= 0) + max(nrows, ncols)*(maxval(labels) - minval(labels) + 1), 2))

        high_edges_buf = 0
        nedges = 0
        call mask2ij( &
            is_high_edge, nrows, ncols, &
            high_edges_buf, size(high_edges_buf, dim=1), nedges)
        ! No high edges found, set z to zero and exit
        if (nedges == 0) then
            z = 0
            return
        end if
        nedges = nedges + 1
        high_edges_buf(nedges, :) = marker

        nlabels = maxval(labels)
        allocate (zmax(nlabels))
        zmax = 0

        ! Initialise z to zero
        z = 0
        queued = .false.
        added_since_marker = .false.

        ! Mark initial seeds as queued
        do iedge = 1, nedges - 1
            ci = high_edges_buf(iedge, 1)
            cj = high_edges_buf(iedge, 2)
            queued(ci, cj) = .true.
        end do
        ! Loop through all high_edges to find cells flowing away from flats
        nloops = 1
        iedge = 1
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
          print *, "[AWAY_FROM_HIGH] Error: High edges buffer overflow (size:", nedges, ", allocated:", size(high_edges_buf, 1), ")"
                    stop
                end if
                high_edges_buf(nedges, :) = marker
                added_since_marker = .false.
                cycle
            end if

            ! Check bounds after marker check
            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
                print *, "[AWAY_FROM_HIGH] Error: Current index out of bounds (", ci, ",", cj, ")"
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
          print *, "[AWAY_FROM_HIGH] Error: High edges buffer overflow (size:", nedges, ", allocated:", size(high_edges_buf, 1), ")"
                    stop
                end if
                high_edges_buf(nedges, :) = [ni, nj]
                queued(ni, nj) = .true.
                added_since_marker = .true.
            end do
        end do
        deallocate (high_edges_buf)
        deallocate (queued)

        ! Adjust z values within flats to ensure they flow away from high edges
        do concurrent(ci=1:nrows, cj=1:ncols, labels(ci, cj) /= 0)
            z(ci, cj) = zmax(labels(ci, cj)) - z(ci, cj) + 1
        end do
        deallocate (zmax)
    end subroutine away_from_high

    subroutine towards_low( &
        z, labels, nrows, ncols, &
        is_low_edge, offsets, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer, intent(in) :: labels(nrows, ncols)
        logical*1, intent(in) :: is_low_edge(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer, intent(out) :: z(nrows, ncols)

        integer, parameter :: marker(2) = [-1, -1]
        logical*1 :: added_since_marker
        integer :: iedge, jedge, nedges, nloops ! Index for low_edges TODO: iedge and jedge can be combined?
        integer :: iofs ! Offset index
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        logical*1, allocatable :: queued(:, :)
        integer, allocatable :: low_edges_buf(:, :)

        allocate (queued(nrows, ncols))
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
        iedge = 1
        nloops = 1
        added_since_marker = .false.
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
               print *, "[TOWARDS_LOW] Error: Low edges buffer overflow (size:", nedges, ", allocated:", size(low_edges_buf, 1), ")"
                    stop
                end if
                low_edges_buf(nedges, :) = marker
                added_since_marker = .false.
                cycle
            end if

            ! Check bounds after marker check
            if (ci < 1 .or. ci > nrows .or. cj < 1 .or. cj > ncols) then
                print *, "[TOWARDS_LOW] Error: Current indices out of bounds (", ci, ",", cj, ")"
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
               print *, "[TOWARDS_LOW] Error: Low edges buffer overflow (size:", nedges, ", allocated:", size(low_edges_buf, 1), ")"
                    stop
                end if
                low_edges_buf(nedges, :) = [ni, nj]
                queued(ni, nj) = .true.
                added_since_marker = .true.
            end do
        end do
        deallocate (queued)
        deallocate (low_edges_buf)
    end subroutine towards_low

    subroutine compute_indegree( &
        flowdir, indegree, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer*1, intent(out) :: indegree(nrows, ncols)

        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer*1 :: code
        integer, allocatable :: diffs(:, :) ! Lookup tables for offsets

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

        indegree = 0

        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj) &
        !$omp COLLAPSE(2) &
        !$omp SCHEDULE(STATIC)
        do ci = 1, nrows
            do cj = 1, ncols
                ! Get neighbour indices based on flow direction
                code = flowdir(ci, cj)
                ni = ci + diffs(code, 1)
                nj = cj + diffs(code, 2)

                ! Check bounds
                if (ni < 1 .or. ni > nrows .or. nj < 1 .or. nj > ncols) cycle
                ! Skip self-loops
                if (ni == ci .and. nj == cj) cycle

                ! Increment indegree of downstream cell, make sure only one thread updates at a time
                !$omp ATOMIC UPDATE
                indegree(ni, nj) = indegree(ni, nj) + int(1, kind=1)
                !$omp END ATOMIC
            end do
        end do
        !$omp END PARALLEL DO
        deallocate (diffs)
    end subroutine compute_indegree

    subroutine compute_accumulation( &
        flowdir, valids, weights, indegrees, accumulations, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valids(nrows, ncols)
        real, intent(in) :: weights(nrows, ncols)
        integer*1, intent(inout) :: indegrees(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        real, intent(out) :: accumulations(nrows, ncols)

        integer :: itofill, ntofills
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer*1 :: code
        integer, allocatable :: diffs(:, :), tofill_buf(:, :) ! Lookup tables for offsets
        logical*1, allocatable :: is_tofill_seed(:, :)

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

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
            indegrees(ni, nj) = indegrees(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegrees(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > size(tofill_buf, 1)) then
          print *, "[COMPUTE_ACCUMULATION] Error: tofill buffer overflow (size:", ntofills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (diffs)
        deallocate (tofill_buf)
    end subroutine compute_accumulation

    subroutine compute_l1_distance( &
        flowdir, valids, indegrees, dists, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valids(nrows, ncols)
        integer*1, intent(inout) :: indegrees(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer, intent(out) :: dists(nrows, ncols)

        integer*1 :: code
        integer :: itofill, ntofills
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        logical*1, allocatable :: is_tofill_seed(:, :)
        integer, allocatable :: tofill_buf(:, :), diffs(:, :)

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero indegree
        allocate (tofill_buf(nrows*ncols, 2))
        allocate (is_tofill_seed(nrows, ncols))
        is_tofill_seed = valids .and. (indegrees == 0)
        call mask2ij(is_tofill_seed, &
                     nrows, ncols, &
                     tofill_buf, nrows*ncols, ntofills)
        deallocate (is_tofill_seed)

        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_buf(itofill, 1)
            cj = tofill_buf(itofill, 2)
            itofill = itofill + 1

            code = flowdir(ci, cj)
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

            ! Update distance of downstream cell
            if (dists(ci, cj) + 1 > dists(ni, nj)) then
                dists(ni, nj) = dists(ci, cj) + sum(diffs(flowdir(ci, cj), :))
            end if
            ! Decrement indegree of downstream cell
            indegrees(ni, nj) = indegrees(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegrees(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > size(tofill_buf, 1)) then
           print *, "[COMPUTE_L1_DISTANCE] Error: tofill buffer overflow (size:", ntofills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (diffs)
        deallocate (tofill_buf)
    end subroutine compute_l1_distance

    subroutine compute_distance( &
        flowdir, valids, x, y, indegrees, dists, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valids(nrows, ncols)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
        integer*1, intent(inout) :: indegrees(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        real, intent(out) :: dists(nrows, ncols)

        integer :: itofill, ntofills
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        real :: step_dist
        integer*1 :: code
        logical*1, allocatable :: is_tofill_seed(:, :)
        integer, allocatable :: tofill_buf(:, :), diffs(:, :)

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero indegree
        allocate (tofill_buf(nrows*ncols, 2))
        allocate (is_tofill_seed(nrows, ncols))
        is_tofill_seed = valids .and. (indegrees == 0)
        call mask2ij(is_tofill_seed, &
                     nrows, ncols, &
                     tofill_buf, nrows*ncols, ntofills)
        deallocate (is_tofill_seed)

        dists = 0.0
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_buf(itofill, 1)
            cj = tofill_buf(itofill, 2)
            itofill = itofill + 1

            code = flowdir(ci, cj)
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

            ! Update distance of downstream cell
            step_dist = hypot( &
                        x(ni, nj) - x(ci, cj), &
                        y(ni, nj) - y(ci, cj))
            if (dists(ci, cj) + step_dist > dists(ni, nj)) then
                dists(ni, nj) = dists(ci, cj) + step_dist
            end if
            ! Decrement indegree of downstream cell
            indegrees(ni, nj) = indegrees(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegrees(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > size(tofill_buf, 1)) then
              print *, "[COMPUTE_DISTANCE] Error: tofill buffer overflow (size:", ntofills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (diffs)
        deallocate (tofill_buf)
    end subroutine compute_distance

    subroutine compute_back_distance( &
        dist, flowdir, x, y, valid, nrows, ncols, offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
        logical*1, intent(in) :: valid(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        real, intent(out) :: dist(nrows, ncols)

        integer :: iseed, nseeds, ifill, nfills
        integer :: si, sj, ci, cj, ui, uj ! Seed, current, upstream indices
        integer :: iofs ! Offset index
        integer*1 :: noflow_code
        logical*1, allocatable :: is_seed(:, :)
        integer, allocatable :: seed_buf(:, :), tofill_buf(:, :)

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

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
           print *, "[COMPUTE_BACK_DISTANCE] Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                        stop
                    end if
                    tofill_buf(nfills, :) = [ui, uj]
                    ! Compute distance
                    dist(ui, uj) = dist(ci, cj) + hypot( &
                                   x(ui, uj) - x(ci, cj), &
                                   y(ui, uj) - y(ci, cj))
                end do
            end do
        end do
        !$omp END DO
        deallocate (tofill_buf)
        !$omp END PARALLEL
        deallocate (seed_buf)
    end subroutine compute_back_distance

    subroutine compute_strahler_order( &
        flowdir, valids, indegrees, orders, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valids(nrows, ncols)
        integer*1, intent(inout) :: indegrees(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer*2, intent(out) :: orders(nrows, ncols)

        integer :: itofill, ntofills
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer*1 :: code
        logical*1, allocatable :: is_tofill_seed(:, :)
        integer, allocatable :: tofill_buf(:, :), diffs(:, :)

        ! Create lookup tables for offsets
        allocate (diffs(0:255, 2))
        diffs = fill_offset_lookup(offsets, codes, noffsets)

        ! Fill the tofill buffer with all valid cells with zero indegree
        allocate (tofill_buf(nrows*ncols, 2))
        allocate (is_tofill_seed(nrows, ncols))
        is_tofill_seed = valids .and. (indegrees == 0)
        call mask2ij(is_tofill_seed, &
                     nrows, ncols, &
                     tofill_buf, nrows*ncols, ntofills)
        deallocate (is_tofill_seed)

        orders = 1
        itofill = 1
        do while (itofill <= ntofills)
            ci = tofill_buf(itofill, 1)
            cj = tofill_buf(itofill, 2)
            itofill = itofill + 1

            code = flowdir(ci, cj)
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

            ! Update distance of downstream cell
            if (orders(ni, nj) < orders(ci, cj)) then
                orders(ni, nj) = orders(ci, cj)
            else if (orders(ni, nj) == orders(ci, cj)) then
                orders(ni, nj) = orders(ni, nj) + int(1, kind=2)
            end if
            ! Decrement indegree of downstream cell
            indegrees(ni, nj) = indegrees(ni, nj) - int(1, kind=1)
            ! If indegree is zero, add to tofill buffer
            if (indegrees(ni, nj) == 0) then
                ntofills = ntofills + 1
                if (ntofills > size(tofill_buf, 1)) then
        print *, "[COMPUTE_STRAHLER_ORDER] Error: tofill buffer overflow (size:", ntofills, ", allocated:", size(tofill_buf, 1), ")"
                    stop
                end if
                tofill_buf(ntofills, :) = [ni, nj]
            end if
        end do
        deallocate (diffs)
        deallocate (tofill_buf)
    end subroutine compute_strahler_order

    subroutine label_watersheds( &
        labels, flowdir, valid, nrows, ncols, offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valid(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        integer, intent(out) :: labels(nrows, ncols)

        integer :: iseed, nseeds, ifill, nfills
        integer :: si, sj, ci, cj, ui, uj ! Seed, current, upstream indices
        integer :: iofs ! Offset index
        integer*1 :: noflow_code
        logical*1, allocatable :: is_seed(:, :)
        integer, allocatable :: seed_buf(:, :), tofill_buf(:, :)

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        labels = 0

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
            labels(si, sj) = iseed
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
                    if (labels(ui, uj) > 0) cycle
                    ! Check if flows into current cell
                    if (flowdir(ui, uj) /= codes(iofs)) cycle

                    ! Add to buffer
                    nfills = nfills + 1
                    if (nfills > size(tofill_buf, 1)) then
                print *, "[LABEL_WATERSHEDS] Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                        stop
                    end if
                    tofill_buf(nfills, :) = [ui, uj]
                    ! Compute distance
                    labels(ui, uj) = labels(ci, cj)
                end do
            end do
        end do
        !$omp END DO
        deallocate (tofill_buf)
        !$omp END PARALLEL
    end subroutine label_watersheds

    subroutine flood_upstream( &
        flooded, flowdir, seeds, valid, nrows, ncols, offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdir(nrows, ncols), codes(noffsets)
        logical*1, intent(in) :: valid(nrows, ncols), seeds(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        logical*1, intent(out) :: flooded(nrows, ncols)

        integer :: iseed, nseeds, ifill, nfills, iofs
        integer :: si, sj, ci, cj, ui, uj ! Seed, current, upstream indices
        integer*1 :: noflow_code
        integer, allocatable :: seed_buf(:, :), tofill_buf(:, :)

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        flooded = .false.

        ! Append all cells with noflow direction to buffer
        allocate (seed_buf(nrows*ncols, 2))
        call mask2ij(seeds, nrows, ncols, &
                     seed_buf, nrows*ncols, nseeds)

        ! Loop through seeds
        !$omp PARALLEL DEFAULT(SHARED) PRIVATE(iseed, si, sj, ci, cj, ifill, nfills, tofill_buf)
        allocate (tofill_buf(nrows*ncols, 2))
        !$omp DO SCHEDULE(DYNAMIC)
        do iseed = 1, nseeds
            si = seed_buf(iseed, 1)
            sj = seed_buf(iseed, 2)

            ! Check if is valid
            if (.not. valid(si, sj)) cycle

            ! Loop through buffer
            nfills = 1
            ifill = 1
            flooded(si, sj) = .true.
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
                    if (flooded(ui, uj)) cycle
                    ! Check if flows into current cell
                    if (flowdir(ui, uj) /= codes(iofs)) cycle

                    ! Add to buffer
                    nfills = nfills + 1
                    if (nfills > size(tofill_buf, 1)) then
                  print *, "[FLOOD_UPSTREAM] Error: tofill buffer overflow (size:", nfills, ", allocated:", size(tofill_buf, 1), ")"
                        stop
                    end if
                    tofill_buf(nfills, :) = [ui, uj]
                    ! Compute distance
                    flooded(ui, uj) = .true.
                end do
            end do
        end do
        !$omp END DO
        deallocate (seed_buf)
        deallocate (tofill_buf)
        !$omp END PARALLEL
    end subroutine flood_upstream

    subroutine compute_max_branch_dist( &
        maxbdists, flowdirs, valids, x, y, basin_ids, nrows, ncols, &
        offsets, codes, noffsets)
        implicit none
        ! Inputs
        integer, intent(in) :: nrows, ncols, noffsets ! Size of the grid and number of offsets
        integer*1, intent(in) :: flowdirs(nrows, ncols), codes(noffsets)
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
        logical*1, intent(in) :: valids(nrows, ncols)
        integer, intent(in) :: basin_ids(nrows, ncols)
        integer, intent(in) :: offsets(noffsets, 2)
        ! Outputs
        real, intent(out) :: maxbdists(nrows, ncols)
        ! Local variables
        real :: dists(2)
        integer :: nneighbour, neighbour_offsets(4, 2)
        integer :: ci, cj, ni, nj ! Current and neighbour indices
        integer, allocatable :: diffs(:, :)
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
                        dists, ci, cj, ni, nj, flowdirs, x, y, diffs, &
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
        s1ij, s2ij, flowdirs, x, y, &
        offset_lookup, check_flag)
        implicit none
        ! Inputs
        integer, intent(in) :: s1ij(2), s2ij(2) ! Indices of the two seed cells
        integer*1, intent(in) :: flowdirs(:, :)
        real, intent(in) :: x(:, :), y(:, :)
        integer, intent(in) :: offset_lookup(0:255, 2)
        logical*1, intent(in), optional :: check_flag ! Whether to check for confluence at each step
        ! Outputs
        real, intent(out) :: dists(2)
        ! Local variables
        logical*1 :: check_flag_
        integer :: maxpathlen
        integer :: id1, id2
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)

        maxpathlen = 4*(size(flowdirs, 1) + size(flowdirs, 2))
        id1 = 1
        id2 = 2
        allocate (path1(maxpathlen, 2))
        allocate (path2(maxpathlen, 2))
        allocate (visited(size(flowdirs, 1), size(flowdirs, 2)))
        visited = 0

        check_flag_ = (.not. present(check_flag)) .or. check_flag

        call inner_compute_confluence_dist( &
            dists, &
            s1ij(1), s1ij(2), s2ij(1), s2ij(2), flowdirs, x, y, offset_lookup, &
            maxpathlen, path1, path2, visited, id1, id2, &
            check_flag=check_flag_)
        deallocate (path1)
        deallocate (path2)
        deallocate (visited)
    end subroutine compute_confluence_dist

    subroutine inner_compute_confluence_dist( &
        dists, s1i, s1j, s2i, s2j, flowdirs, x, y, &
        offset_lookup, maxpathlen, path1, path2, visited, id1, id2, check_flag)
        implicit none
        ! Inputs
        integer, intent(in) :: s1i, s1j, s2i, s2j ! Indices of the two seed cells
        integer*1, intent(in) :: flowdirs(:, :)
        real, intent(in) :: x(:, :), y(:, :)
        integer, intent(in) :: offset_lookup(0:255, 2)
        logical*1, intent(in), optional :: check_flag ! Whether to check for confluence at each step
        integer, intent(in) :: maxpathlen
        integer, intent(inout) :: path1(maxpathlen, 2), path2(maxpathlen, 2) ! Indices of paths
        integer :: id1, id2
        integer, intent(inout) :: visited(:, :) ! A grid to track visited paths by ids
        ! Outputs
        real :: dists(2)
        ! Local variables
        integer :: ipath1, ipath2, npath1, npath2 ! Lengths of paths
        integer :: iconf1, iconf2 ! Indices of confluence in paths
        integer :: n1i, n1j, n2i, n2j
        integer*1 :: code1, code2
        logical*1 :: is_active1, is_active2, local_check_flag

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
                code1 = flowdirs(path1(npath1, 1), path1(npath1, 2))
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
                ! print *, "Path1 step from ", path1(npath1, 1), ",", path1(npath1, 2), " to ", n1i, ",", n1j, "(code ", code1, ")"
                if (n1i < 1 .or. n1i > size(flowdirs, 1) .or. n1j < 1 .or. n1j > size(flowdirs, 2)) then
                    iconf1 = npath1
                    is_active1 = .false.
                    ! print *, "Path1 out of bounds"
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
                    ! print *, "Confluence found at ", n1i, ",", n1j
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
                code2 = flowdirs(path2(npath2, 1), path2(npath2, 2))
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
                ! print *, "Path2 step from ", path2(npath2, 1), ",", path2(npath2, 2), " to ", n2i, ",", n2j, "(code ", code2, ")"
                if (n2i < 1 .or. n2i > size(flowdirs, 1) .or. n2j < 1 .or. n2j > size(flowdirs, 2)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    ! print *, "Path2 out of bounds"
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
                    ! print *, "Confluence found at ", n2i, ",", n2j
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
    !     integer*1, allocatable :: indegrees(:, :)
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
    !     call compute_l1_distance( &
    !         flowdirs, valids, indegrees, dists, nrows, ncols, &
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
