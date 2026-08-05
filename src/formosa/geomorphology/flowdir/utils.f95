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
!   2026-08-03, En-Chi Lee (williameclee@gmail.com)
!     - Explicitly handled Python uint8 -> signed 8-bit Fortran conversion/interpretation in 'fill_offset_lookup'
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Refactored 'mask2ij' to propagate buffer overflow error
!     - Added function 'mask2id' as the linear-index version of 'mask2ij'; also added related linear index check functions
!   2026-08-05, En-Chi Lee (williameclee@gmail.com)
!     - Switched to 'iso_c_binding'
!!!

module utils
    use iso_c_binding, only: c_int8_t
    implicit none
contains
    pure function ij2id_checked(i, j, nrows, ncols) result(cell_id)
        !! Encodes a valid one-based grid coordinate as a linear cell ID.
        !! Zero is returned for an out-of-bounds coordinate and is never a
        !! valid cell ID.
        implicit none
        integer, intent(in) :: i, j, nrows, ncols
        integer :: cell_id

        if (nrows < 1 .or. ncols < 1) then
            cell_id = 0
        else if (ncols > huge(cell_id)/nrows) then
            cell_id = 0
        else if (i < 1 .or. i > nrows .or. j < 1 .or. j > ncols) then
            cell_id = 0
        else
            cell_id = i + (j - 1)*nrows
        end if
    end function ij2id_checked

    pure subroutine id2ij_checked(cell_id, nrows, ncols, i, j, is_valid)
        !! Decodes a one-based linear cell ID, rejecting IDs outside the grid.
        implicit none
        integer, intent(in) :: cell_id, nrows, ncols
        integer, intent(out) :: i, j
        logical(kind=1), intent(out) :: is_valid

        is_valid = nrows >= 1 .and. ncols >= 1
        if (is_valid) is_valid = ncols <= huge(cell_id)/nrows
        if (is_valid) is_valid = cell_id >= 1 .and. cell_id <= nrows*ncols
        if (.not. is_valid) then
            i = 0
            j = 0
            return
        end if
        i = mod(cell_id - 1, nrows) + 1
        j = (cell_id - 1)/nrows + 1
    end subroutine id2ij_checked

    pure subroutine mask2id(mask, ids, nids, cnt, err_code)
        !! Converts a mask to validated one-based linear cell IDs.
        implicit none
        logical(kind=1), intent(in) :: mask(:, :)
        integer, intent(in) :: nids
        integer, intent(out) :: ids(nids), cnt, err_code
        integer :: ci, cj

        cnt = 0
        err_code = 0
        do cj = lbound(mask, 2), ubound(mask, 2)
            do ci = lbound(mask, 1), ubound(mask, 1)
                if (.not. mask(ci, cj)) cycle
                if (cnt == nids) then
                    err_code = 3
                    return
                end if
                cnt = cnt + 1
                ids(cnt) = ij2id_checked(ci, cj, size(mask, 1), size(mask, 2))
            end do
        end do
    end subroutine mask2id

    pure subroutine mask2ij(mask, ij, nij, cnt, err_code)
        !! Converts a 2D logical mask to a list of (i, j) indices where
        !! the mask is true.
        !! The output list will have a maximum size of 2-by-'nij', and
        !! the actual number of valid indices found will be returned in
        !! 'cnt'. If the number of valid indices exceeds nij, the
        !! remaining will be ignored.
        implicit none
        ! Arguments
        logical(kind=1), intent(in) :: mask(:, :)
            !! Input logical mask
        integer, intent(in) :: nij
            !! Maximum number of indices to return
        ! Outputs
        integer, intent(out) :: ij(2, nij)
            !! Output list of (i, j) indices where mask is true, with a maximum size of 2-by-nij
        integer, intent(out) :: cnt
            !! Actual number of valid indices found (up to nij)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 3: Output index buffer capacity was exceeded
        ! Local variables
        integer :: ci, cj

        ! Count number of valid neighbors
        cnt = 0
        err_code = 0

        do cj = lbound(mask, 2), ubound(mask, 2)
            do ci = lbound(mask, 1), ubound(mask, 1)
                if (.not. mask(ci, cj)) cycle
                if (cnt == nij) then
                    err_code = 3
                    return
                end if
                cnt = cnt + 1
                ij(1, cnt) = ci
                ij(2, cnt) = cj
            end do
        end do
    end subroutine mask2ij

    pure function find_noflow_code(offsets, codes, default_noflow_code) result(noflow_code)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, find the code that corresponds to the no-flow
        !! direction (0, 0). If not found, return the provided default
        !! no-flow code or 0 if not provided.
        implicit none
        ! Arguments
        integer, intent(in) :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in) :: codes(:)
            !! List of codes corresponding to the offsets
        integer(c_int8_t), intent(in), optional :: default_noflow_code
            !! Optional default no-flow code to use if not found in offsets (default: 0)
        integer(c_int8_t) :: noflow_code
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
        do iofs = 1, size(codes)
            if (offsets(iofs, 1) == 0 .and. offsets(iofs, 2) == 0) then
                noflow_code = codes(iofs)
                exit
            end if
        end do
    end function find_noflow_code

    pure function find_opposite_codes(offsets, codes) result(opp_codes)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, find the list of codes that correspond to the
        !! opposite direction of each code.
        !! For example, if code 1 corresponds to offset (1, 0), and code
        !! 2 corresponds to offset (-1, 0), then code 2 is the opposite
        !! code of code 1 and vice verse.
        implicit none
        ! Arguments
        integer, intent(in) :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in) :: codes(:)
            !! List of codes corresponding to the offsets
        integer(c_int8_t) :: opp_codes(size(codes))
            !! List of opposite codes corresponding to the offsets (same order as input codes)
        ! Local variables
        integer :: iofs, jofs
            !! Offset indices for iterating

        ! Loop through offsets to find opposite codes
        do iofs = 1, size(codes)
            do jofs = 1, size(codes)
                if (offsets(iofs, 1) == -offsets(jofs, 1) .and. &
                    offsets(iofs, 2) == -offsets(jofs, 2)) then
                    opp_codes(iofs) = codes(jofs)
                    exit
                end if
            end do
        end do
    end function find_opposite_codes

    pure function fill_offset_lookup(offsets, codes) result(diffs)
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
        integer, intent(in) :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in) :: codes(:)
            !! List of codes corresponding to the offsets
        ! Outputs
        integer :: diffs(0:255, 2)
            !! Lookup table for offsets
        ! Local variables
        integer :: iofs, code

        ! Create lookup tables for offsets
        diffs = -99 ! Initialise to invalid value
        do iofs = 1, size(codes)
            ! F2PY exposes uint8 arrays as signed 8-bit integer values. Preserve
            ! their bit pattern so codes 128:255 address the intended slots.
            code = iand(int(codes(iofs)), 255)
            ! Fill in the offset for the corresponding code index
            diffs(code, 1) = offsets(iofs, 1)
            diffs(code, 2) = offsets(iofs, 2)
        end do
    end function fill_offset_lookup
end module utils
