!!!
! Last modified
!   2026-07-08, En-Chi Lee (williameclee@gmail.com)
!     - Moved distance calculation to this module
!   2026-07-09, En-Chi Lee (williameclee@gmail.com)
!     - Implemented point-to-line distance functions 'pt2linedist_xy' and 'pt2linedist2_xy'
!   2026-07-10, En-Chi Lee (williameclee@gmail.com)
!     - Implemented 2D line intersection test function 'lines_intersect_v2'
!!!

module distances
    implicit none
    interface l1dist_xy
        module procedure l1dist_xy_int
        module procedure l1dist_xy_real
    end interface l1dist_xy
contains
    pure function l1dist_xy_int(x1, y1, x2, y2) result(dist)
        !! Calculates the L1 distance between two 2D points.
        implicit none
        ! Arguments
        integer, intent(in) :: x1, y1, x2, y2
        integer :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_int

    pure function l1dist_xy_real(x1, y1, x2, y2) result(dist)
        !! Calculates the L1 distance between two 2D points.
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_real

    pure function l2dist_xy(x1, y1, x2, y2) result(dist)
        !! Calculates the L2 distance between two 2D points.
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        ! 'sqrt' is unsed instead of 'hypot' because coordinates are bounded
        ! and do not present underflow/overflow risk here.
        dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    end function l2dist_xy

    pure function pt2linedist2_xy(x1, y1, x2, y2, x3, y3) result(dist2)
        ! Calculates the squared distance of a point to a line segment defined by two points on a 2D plane.
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
            !! x and y coordinates of the two endpoints of the line segment
        real, intent(in) :: x3, y3
            !! x and y coordinates of the point to be measured
        ! Outputs
        real :: dist2
            !! Squared distance of the point to the line segment
        ! Local variables
        real :: lx, ly, px, py
            !! x and y components of the vectors from an end point to the other and to the out-of-line point, respectively
        real :: t
            !! Helper variable to track the relative position of the point projected on to the line segment

        lx = x2 - x1
        ly = y2 - y1
        px = x3 - x1
        py = y3 - y1

        t = (lx*px + ly*py)/(lx**2 + ly**2)
        t = min(max(t, 0.), 1.)
        dist2 = (x3 - (x1 + lx*t))**2 + (y3 - (y1 + ly*t))**2
    end function pt2linedist2_xy

    pure function pt2linedist_xy(x1, y1, x2, y2, x3, y3) result(dist)
        ! Calculates the distance of a point to a line segment defined by two points on a 2D plane.
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
            !! x and y coordinates of the two endpoints of the line segment
        real, intent(in) :: x3, y3
            !! x and y coordinates of the point to be measured
        ! Outputs
        real :: dist
            !! Distance of the point to the line segment

        dist = sqrt(pt2linedist2_xy(x1, y1, x2, y2, x3, y3))
    end function pt2linedist_xy

    pure function bboxes_overlap(p1, p2, p3, p4) result(flag_overlap)
        implicit none
        ! Arguments
        real, intent(in) :: p1(2), p2(2), p3(2), p4(2)
        ! Outputs
        logical*1 :: flag_overlap

        flag_overlap = &
            (max(min(p1(1), p2(1)), min(p3(1), p4(1))) <= min(max(p1(1), p2(1)), max(p3(1), p4(1)))) .and. &
            (max(min(p1(2), p2(2)), min(p3(2), p4(2))) <= min(max(p1(2), p2(2)), max(p3(2), p4(2))))
    end function bboxes_overlap

    pure function lines_intersect_v2(l1a, l1b, l2a, l2b) result(flag)
        !! Flags:
        !! -1 : disjoint
        !!  0 : endpoint-to-endpoint touch
        !!  1 : interior-interior crossing (X)
        !!  2 : collinear overlap, not identical
        !!  3 : identical segment
        !!  4 : endpoint-on-interior (T-junction)
        !!  5 : degenerate segment (some line is actually a point)
        implicit none
        ! Arguments
        real, intent(in) :: l1a(2), l1b(2), l2a(2), l2b(2)
        ! Outputs
        integer*1 :: flag
        ! Local variables
        logical*1 :: eq_l1al2a, eq_l1al2b, eq_l1bl2a, eq_l1bl2b
        integer*1 :: o1, o2, o3, o4
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
    contains
        pure function orient_v2(p1, p2, p3) result(o)
            implicit none
            ! Arguments
            real, intent(in) :: p1(2), p2(2), p3(2)
            ! Outputs
            integer*1 :: o
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
            implicit none
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
    end function lines_intersect_v2

end module distances
