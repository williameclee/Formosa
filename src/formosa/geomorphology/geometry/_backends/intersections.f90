!> Classify line-segment intersections using the FORTRAN backend.
!!
!! This internal module is called by the Python geometry API and
!! other FORTRAN routines and is not intended to be used directly.
!!
!! Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
module intersections
    use iso_c_binding, only: c_int8_t
    implicit none(type, external)
contains
    pure function bboxes_overlap(p1, p2, p3, p4) result(flag_overlap)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: p1(2), p2(2), p3(2), p4(2)
        ! Outputs
        logical(kind=1) :: flag_overlap

        flag_overlap = &
            (max(min(p1(1), p2(1)), min(p3(1), p4(1))) <= min(max(p1(1), p2(1)), max(p3(1), p4(1)))) .and. &
            (max(min(p1(2), p2(2)), min(p3(2), p4(2))) <= min(max(p1(2), p2(2)), max(p3(2), p4(2))))
    end function bboxes_overlap

    pure function orient_v2(p1, p2, p3) result(o)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: p1(2), p2(2), p3(2)
        ! Outputs
        integer(c_int8_t) :: o
        ! Local variables
        real :: xprod

        xprod = (p2(1) - p1(1))*(p3(2) - p1(2)) - (p2(2) - p1(2))*(p3(1) - p1(1))

        if (xprod == 0) then
            o = 0
        else if (xprod < 0) then
            o = -1
        else
            o = 1
        end if
    end function orient_v2

    pure function on_segment(a, b, p) result(on_flag)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: a(2), b(2), p(2)
        ! Outputs
        logical :: on_flag

        if (orient_v2(a, b, p) /= 0) then
            on_flag = .false.
            return
        end if

        on_flag = p(1) >= min(a(1), b(1)) .and. &
                  p(1) <= max(a(1), b(1)) .and. &
                  p(2) >= min(a(2), b(2)) .and. &
                  p(2) <= max(a(2), b(2))
    end function on_segment

    pure function lines_intersect_v2(l1a, l1b, l2a, l2b) result(flag)
        !! Flags:
        !! -1 : disjoint
        !!  0 : endpoint-to-endpoint touch
        !!  1 : interior-interior crossing (X)
        !!  2 : collinear overlap, not identical
        !!  3 : identical segment
        !!  4 : endpoint-on-interior (T-junction)
        !!  5 : degenerate segment (some line is actually a point)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: l1a(2), l1b(2), l2a(2), l2b(2)
        ! Outputs
        integer(c_int8_t) :: flag
        ! Local variables
        logical(kind=1) :: eq_l1al2a, eq_l1al2b, eq_l1bl2a, eq_l1bl2b
        integer(c_int8_t) :: o1, o2, o3, o4
        real :: a0, a1, c0, c1, tmp, overlap0, overlap1

        flag = -1

        if (all(l1a == l1b) .or. all(l2a == l2b)) then ! Degeneracy test
            flag = 5
            return
        else if (.not. bboxes_overlap(l1a, l1b, l2a, l2b)) then ! Bounding box intersection test
            flag = -1
            return
        end if

        eq_l1al2a = all(l1a == l2a)
        eq_l1al2b = all(l1a == l2b)
        eq_l1bl2a = all(l1b == l2a)
        eq_l1bl2b = all(l1b == l2b)
        if ((eq_l1al2a .and. eq_l1bl2b) .or. (eq_l1al2b .and. eq_l1bl2a)) then ! Same line test
            flag = 3
            return
        end if

        o1 = orient_v2(l1a, l1b, l2a)
        o2 = orient_v2(l1a, l1b, l2b)
        o3 = orient_v2(l2a, l2b, l1a)
        o4 = orient_v2(l2a, l2b, l1b)
        if ((o1*o2 < 0) .and. (o3*o4) < 0) then ! Interior-interior crossing
            flag = 1
            return
        end if

        ! Collinear case.
        if ((o1 == 0) .and. (o2 == 0) .and. (o3 == 0) .and. (o4 == 0)) then
            ! Project onto the dominant axis.
            if (abs(l1b(1) - l1a(1)) >= abs(l1b(2) - l1a(2))) then
                a0 = l1a(1)
                a1 = l1b(1)
                c0 = l2a(1)
                c1 = l2b(1)
            else
                a0 = l1a(2)
                a1 = l1b(2)
                c0 = l2a(2)
                c1 = l2b(2)
            end if

            if (a0 > a1) then
                tmp = a0
                a0 = a1
                a1 = tmp
            end if
            if (c0 > c1) then
                tmp = c0
                c0 = c1
                c1 = tmp
            end if

            overlap0 = max(a0, c0)
            overlap1 = min(a1, c1)

            if (overlap1 < overlap0) then
                flag = -1
            else if (overlap1 <= overlap0) then ! They touch at exactly one endpoint
                flag = 0
            else ! They overlap over a nonzero interval
                flag = 2
            end if
            return
        end if

        ! Endpoint-to-endpoint touch
        if (eq_l1al2a .or. eq_l1al2b .or. eq_l1bl2a .or. eq_l1bl2b) then
            flag = 0
            return
        end if

        ! T-junction
        if (on_segment(l1a, l1b, l2a) .or. &
            on_segment(l1a, l1b, l2b) .or. &
            on_segment(l2a, l2b, l1a) .or. &
            on_segment(l2a, l2b, l1b)) then
            flag = 4
        end if
    end function lines_intersect_v2

end module intersections
