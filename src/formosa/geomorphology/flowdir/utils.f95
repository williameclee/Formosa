!!!
! Last modified
!   2026-07-02, En-Chi Lee (williameclee@gmail.com)
!     - Iterated the array bound instead of starting from 1 in 'mask2ij'
!   2026-07-08, En-Chi Lee (williameclee@gmail.com)
!     - Moved 'mask2ij' to this module
!   2026-07-08, En-Chi Lee (williameclee@gmail.com)
!     - Moved 'mask2ij' to separate 'utils' module
!   2026-07-12, En-Chi Lee (williameclee@gmail.com)
!     - Moved 'flowdir_utils' module here 'utils'
!!!

module utils
    implicit none
contains
    subroutine mask2ij( &
        mask, nrows, ncols, ij, nij, cnt)
        !! Converts a 2D logical mask to a list of (i, j) indices where
        !! the mask is true.
        !! The output list will have a maximum size of 2-by-'nij', and
        !! the actual number of valid indices found will be returned in
        !! 'cnt'. If the number of valid indices exceeds nij, the
        !! remaining will be ignored.
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the input mask
        logical*1, intent(in) :: mask(nrows, ncols)
            !! Input logical mask
        integer, intent(in) :: nij
            !! Maximum number of indices to return
        ! Outputs
        integer, intent(out) :: ij(2, nij)
            !! Output list of (i, j) indices where mask is true, with a maximum size of 2-by-nij
        integer, intent(out) :: cnt
            !! Actual number of valid indices found (up to nij)
        ! Local variables
        integer :: ci, cj

        ! Count number of valid neighbors
        cnt = 0

        do cj = lbound(mask, 2), ubound(mask, 2)
            do ci = lbound(mask, 1), ubound(mask, 1)
                if (.not. mask(ci, cj)) cycle
                if (cnt == nij) then
                    print *, "Warning: mask2ij found more valid indices than the maximum allowed (", cnt, "). Only the first ", nij, " indices will be returned."
                    return
                end if
                cnt = cnt + 1
                ij(1, cnt) = ci
                ij(2, cnt) = cj
            end do
        end do
    end subroutine mask2ij

    pure function find_noflow_code(offsets, codes, noffsets, default_noflow_code) result(noflow_code)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, find the code that corresponds to the no-flow
        !! direction (0, 0). If not found, return the provided default
        !! no-flow code or 0 if not provided.
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

    pure function find_opposite_codes(offsets, codes, noffsets) result(opp_codes)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, find the list of codes that correspond to the
        !! opposite direction of each code.
        !! For example, if code 1 corresponds to offset (1, 0), and code
        !! 2 corresponds to offset (-1, 0), then code 2 is the opposite
        !! code of code 1 and vice verse.
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

    pure function fill_offset_lookup(offsets, codes, noffsets) result(diffs)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, create a lookup table (array) where the index
        !! corresponds to the code and the value is the offset.
        !! The offset codes must be between 0 and 255, and the returned
        !! lookup table will have a size of 256-by-2 to accommodate all
        !! possible codes. Unused indices will have an offset of
        !! (-99, -99) to indicate invalid code.
        !! For example, if code 1 corresponds to offset (1, 0), then
        !! diffs(1, :) = (1, 0).
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
                error stop "[FILL_OFFSET_LOOKUP] Encountered out-of-bound flow direction code"
            end if
            ! Fill in the offset for the corresponding code index
            diffs(codes(iofs), 1) = offsets(iofs, 1)
            diffs(codes(iofs), 2) = offsets(iofs, 2)
        end do
    end function fill_offset_lookup
end module utils
