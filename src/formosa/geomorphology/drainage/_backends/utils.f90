!> Provides shared utilities for the FORTRAN drainage backends.
!!
!! This internal module supports array-index conversion, flow-
!! direction code decoding, raster masking, and priority queues used
!! by other FORTRAN modules.
!!
!! Last modified: 2026-08-14, En-Chi Lee (williameclee@gmail.com)
module utils
    use iso_c_binding, only: c_int8_t
    implicit none(type, external)

    integer, parameter :: ERR_NO_ERROR = 0
    integer, parameter :: ERR_INVALID_INPUT = 1
    integer, parameter :: ERR_ALLOCATION_FAILURE = 2
    integer, parameter :: ERR_OVERFLOW = 3
    integer, parameter :: ERR_COMPUTATION_FAILURE = 4
contains

    logical pure function array2d_oob(i, j, nrows, ncols) result(is_oob)
        !! Checks whether a pair of array index (i, j) is
        !! out-of-bounds
        implicit none(type, external)
        integer, intent(in) :: i, j
            !! Row and column indices
        integer, intent(in) :: nrows, ncols
            !! Size of the array

        if (i < 1 .or. i > nrows .or. j < 1 .or. j > ncols) then
            is_oob = .true.
        else
            is_oob = .false.
        end if
    end function array2d_oob

    pure function ij2id_checked(i, j, nrows, ncols) result(cell_id)
        !! Encodes a valid one-based grid coordinate as a linear
        !! cell ID.
        !!
        !! Zero is returned for an out-of-bounds coordinate and is
        !! never a valid cell ID.
        implicit none(type, external)
        integer, intent(in) :: i, j
            !! Row and column indices
        integer, intent(in) :: nrows, ncols
            !! Size of the array
        integer :: cell_id

        if (nrows < 1 .or. ncols < 1) then
            cell_id = 0
        else if (ncols > huge(cell_id)/nrows) then
            cell_id = 0
        else if (array2d_oob(i, j, nrows, ncols)) then
            cell_id = 0
        else
            cell_id = i + (j - 1)*nrows
        end if
    end function ij2id_checked

    pure subroutine id2ij_checked(cell_id, nrows, ncols, i, j, is_valid)
        !! Decodes a one-based linear cell ID, rejecting IDs outside the grid.
        implicit none(type, external)
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
        implicit none(type, external)
        logical(kind=1), intent(in), contiguous :: mask(:, :)
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
        implicit none(type, external)
        ! Arguments
        logical(kind=1), intent(in), contiguous :: mask(:, :)
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

    !> Increments ***a*** by ***s*** and wraps it back to the range
    !! [1, ***p***].
    pure function modshift(a, s, p) result(b)
        implicit none(type, external)
        integer, intent(in) :: a, s, p
        integer :: b
        b = modulo(a + s - 1, p) + 1
    end function modshift

    pure function find_noflow_code(offsets, codes, default_noflow_code) result(noflow_code)
        !! For pairs of flow direction codes and their corresponding
        !! offsets, find the code that corresponds to the no-flow
        !! direction (0, 0). If not found, return the provided default
        !! no-flow code or 0 if not provided.
        implicit none(type, external)
        ! Arguments
        integer, intent(in), contiguous :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in), contiguous :: codes(:)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in), contiguous :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in), contiguous :: codes(:)
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
        implicit none(type, external)
        ! Arguments
        integer, intent(in), contiguous :: offsets(:, :)
            !! List of offsets
        integer(c_int8_t), intent(in), contiguous :: codes(:)
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

    pure logical function is_lower_id(id1, id2, z) result(is_lower)
        !! Decides whether a cell is lower than another cell and
        !! therefore has a higher priority in the priority queue.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: id1, id2
            !! Linear cell IDs of the 'z' array to compare
        real, intent(in) :: z(*)
            !! Array the cell IDs reference and used to compare
            !! values

        if (z(id1) < z(id2)) then
            is_lower = .true.
        else
            is_lower = .false.
        end if
    end function is_lower_id

    pure subroutine push_priority_queue( &
        queue, queue_size, new, z, err_code)
        !! Pushes a cell id to the priority queue.
        !!
        !! The cell is moved to an appropriate location such that
        !! the cell's corresponding value in 'z' is larger than its
        !! parent's but smaller than all its children.
        !!
        !! Notes
        !! -----
        !! See :func:'pop_priority_queue' for the push operation.
        implicit none(type, external)
        ! Arguments
        integer, intent(inout) :: queue(:)
            !! Priority queue, containing linear cell IDs of 'z'
        integer, intent(inout) :: queue_size
            !! Total number of cells in the queue (not the size/
            !! capacity of the queue)
        integer, intent(in) :: new
            !! New cell ID to push into the queue
        real, intent(in) :: z(*)
            !! Array the cell IDs reference and used to compare
            !! values
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: Invalid input
            !!   - 3: Queue overflow
        ! Local variables
        integer :: pos, parent_pos
            !! For swapping the new cell to the right position
        integer :: tmp_id
            !! For swapping the new cell to the right position

        err_code = 0

        ! First make sure the queue is large enough
        if (queue_size < 0) then
            err_code = 1 ! Incorrect input
            return
        elseif (queue_size >= size(queue)) then
            err_code = 3 ! Overflow
            return
        end if

        ! Add the new cell
        queue_size = queue_size + 1
        pos = queue_size
        queue(pos) = new

        ! Move the new cell to the right position
        do while (pos > 1)
            parent_pos = pos/2
            ! Swap with parent if current cell is lower
            if (.not. (is_lower_id(queue(pos), queue(parent_pos), z))) exit
            tmp_id = queue(parent_pos)
            queue(parent_pos) = queue(pos)
            queue(pos) = tmp_id
            pos = parent_pos
        end do
    end subroutine push_priority_queue

    pure subroutine pop_priority_queue( &
        queue, queue_size, popped, z, err_code)
        !! Pops the linear cell ID with the highest priority from
        !! the priority queue.
        !!
        !! The queue is then resorted such that the internal tree
        !! structure (such that all children have larger value than
        !! their parent) is preserved.
        !!
        !! Notes
        !! -----
        !! See :func:'push_priority_queue' for the push operation.
        implicit none(type, external)
        ! Arguments
        integer, intent(inout) :: queue(:)
            !! Priority queue, containing linear cell IDs of 'z'
        integer, intent(inout) :: queue_size
            !! Total number of cells in the queue (not the size/
            !! capacity of the queue)
        real, intent(in) :: z(*)
            !! Array the cell IDs reference and used to compare
            !! values
        ! Outputs
        integer, intent(out) :: popped
            !! Linear cell ID of the popped cell
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: Invalid input
            !!   - 3: Queue overflow
        ! Local variables
        integer :: pos, left_pos, right_pos, lower_pos
            !! For swapping the new cell to the right position
        integer :: tmp_id
            !! For swapping the new cell to the right position

        err_code = 0

        ! Check the queue is normal
        if (queue_size <= 0) then
            err_code = 1 ! Incorrect input
            popped = 0
            return
        elseif (queue_size > size(queue)) then
            err_code = 3 ! Overflow
            popped = 0
            return
        end if

        popped = queue(1)
        if (queue_size == 1) then
            queue_size = 0
            return
        end if

        ! Sort the rest of the queue
        ! First move the last element to the top
        pos = 1
        queue(pos) = queue(queue_size)
        queue_size = queue_size - 1
        ! Now move it to the correct place
        left_pos = pos*2
        do while (left_pos <= queue_size)
            right_pos = left_pos + 1

            ! Find the lower of the two child to swap with
            lower_pos = left_pos
            if (right_pos <= queue_size) then
                if (is_lower_id(queue(right_pos), queue(left_pos), z)) &
                    lower_pos = right_pos
            end if

            ! Check if needs swapping
            if (.not. is_lower_id(queue(lower_pos), queue(pos), z)) exit
            ! Swap
            tmp_id = queue(lower_pos)
            queue(lower_pos) = queue(pos)
            queue(pos) = tmp_id

            pos = lower_pos
            left_pos = pos*2
        end do
    end subroutine pop_priority_queue
end module utils
