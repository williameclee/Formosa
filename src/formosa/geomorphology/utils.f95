!!!
! Last modified
!   2026-07-08, En-Chi Lee (williameclee@gmail.com)
!     - Moved 'mask2ij' to this module
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
end module utils
