!> Evaluates geometric predicates using the Fortran backend.
!!
!! This internal module provides orientation and in-circle
!! predicates for triangulation, together with line-segment
!! intersection routines. It is called by the Python geometry API
!! and other Fortran routines and is not intended to be used
!! directly.
!!
!! Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
module intersections
    use iso_c_binding, only: c_double, c_int8_t, c_int32_t, c_int64_t
    implicit none(type, external)
    private :: incircle_double, saturate_int32, saturate_int64
    interface bboxes_overlap
        module procedure bboxes_overlap_int32
        module procedure bboxes_overlap_real
    end interface bboxes_overlap
    interface orient
        module procedure orient_real
        module procedure orient_int32
        module procedure orient_int64
    end interface orient
    interface xcross_orient
        module procedure xcross_orient_real
        module procedure xcross_orient_int32
        module procedure xcross_orient_int64
    end interface xcross_orient
    interface xcross
        module procedure xcross_real
        module procedure xcross_int32
        module procedure xcross_int64
    end interface xcross
    interface incircle
        module procedure incircle_real
        module procedure incircle_int32
        module procedure incircle_int64
    end interface incircle
contains
    pure function bboxes_overlap_int32(p1, p2, p3, p4) result(flag_overlap)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: p1(2), p2(2), p3(2), p4(2)
        ! Outputs
        logical(kind=1) :: flag_overlap

        flag_overlap = &
            (max(min(p1(1), p2(1)), min(p3(1), p4(1))) <= &
             min(max(p1(1), p2(1)), max(p3(1), p4(1)))) .and. &
            (max(min(p1(2), p2(2)), min(p3(2), p4(2))) <= &
             min(max(p1(2), p2(2)), max(p3(2), p4(2))))
    end function bboxes_overlap_int32

    pure function bboxes_overlap_real(p1, p2, p3, p4) result(flag_overlap)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: p1(2), p2(2), p3(2), p4(2)
        ! Outputs
        logical(kind=1) :: flag_overlap

        flag_overlap = &
            (max(min(p1(1), p2(1)), min(p3(1), p4(1))) <= &
             min(max(p1(1), p2(1)), max(p3(1), p4(1)))) .and. &
            (max(min(p1(2), p2(2)), min(p3(2), p4(2))) <= &
             min(max(p1(2), p2(2)), max(p3(2), p4(2))))
    end function bboxes_overlap_real

    pure function saturate_int32(value) result(saturated)
        implicit none(type, external)
        real(c_double), intent(in) :: value
        integer(c_int32_t) :: saturated

        if (value > real(huge(saturated), kind=c_double)) then
            saturated = huge(saturated)
        else if (value < -real(huge(saturated), kind=c_double) - 1.0_c_double) then
            saturated = -huge(saturated) - 1_c_int32_t
        else
            saturated = int(value, kind=c_int32_t)
        end if
    end function saturate_int32

    pure function saturate_int64(value) result(saturated)
        implicit none(type, external)
        real(c_double), intent(in) :: value
        integer(c_int64_t) :: saturated

        ! Converting INT64_MAX to double rounds upward to 2**63, so
        ! use inclusive comparisons before conversion to keep it
        ! defined.
        if (value >= real(huge(saturated), kind=c_double)) then
            saturated = huge(saturated)
        else if (value <= -real(huge(saturated), kind=c_double)) then
            saturated = -huge(saturated) - 1_c_int64_t
        else
            saturated = int(value, kind=c_int64_t)
        end if
    end function saturate_int64

    !> Calculates the signed orientation determinant for real
    !! coordinates.
    !!
    !! A positive result denotes counterclockwise orientation. A
    !! negative result denotes clockwise orientation. Zero denotes
    !! collinearity.
    pure function orient_real(p1, p2, p3) result(o)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: p1(2), p2(2), p3(2)
        ! Outputs
        real :: o

        o = (p2(1) - p1(1))*(p3(2) - p1(2)) - (p2(2) - p1(2))*(p3(1) - p1(1))
    end function orient_real

    !> Calculates the signed orientation determinant for 32-bit
    !! coordinates.
    !!
    !! The calculation uses double-precision intermediates. A result
    !! outside the 32-bit range is saturated to the corresponding
    !! integer limit. The result retains the 32-bit input kind and
    !! preserves the predicate sign.
    pure function orient_int32(p1, p2, p3) result(o)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: p1(2), p2(2), p3(2)
        ! Outputs
        integer(c_int32_t) :: o
        ! Local variables
        real(c_double) :: p1d(2), p2d(2), p3d(2)

        p1d = real(p1, kind=c_double)
        p2d = real(p2, kind=c_double)
        p3d = real(p3, kind=c_double)
        o = saturate_int32((p2d(1) - p1d(1))*(p3d(2) - p1d(2)) - &
                           (p2d(2) - p1d(2))*(p3d(1) - p1d(1)))
    end function orient_int32

    !> Calculates the signed orientation determinant for 64-bit
    !! coordinates.
    !!
    !! The calculation uses double-precision intermediates. A result
    !! outside the 64-bit range is saturated to the corresponding
    !! integer limit. The result retains the 64-bit input kind and
    !! preserves the predicate sign.
    pure function orient_int64(p1, p2, p3) result(o)
        implicit none(type, external)
        ! Arguments
        ! f2py does not currently emit its long_long typedef for
        ! c_int64_t.
        !f2py integer(kind=8), intent(in) :: p1, p2, p3
        integer(c_int64_t), intent(in) :: p1(2), p2(2), p3(2)
        ! Outputs
        !f2py integer(kind=8) :: o
        integer(c_int64_t) :: o
        ! Local variables
        real(c_double) :: p1d(2), p2d(2), p3d(2)

        p1d = real(p1, kind=c_double)
        p2d = real(p2, kind=c_double)
        p3d = real(p3, kind=c_double)
        o = saturate_int64((p2d(1) - p1d(1))*(p3d(2) - p1d(2)) - &
                           (p2d(2) - p1d(2))*(p3d(1) - p1d(1)))
    end function orient_int64

    !> Calculates the signed in-circle determinant for real
    !! coordinates.
    !!
    !! For counterclockwise triangle vertices, a positive result
    !! places the test point inside their circumcircle. Zero places
    !! the point on the circle, and a negative result places it
    !! outside.
    pure function incircle_real(a, b, c, p) result(det)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: a(2), b(2), c(2), p(2)
        ! Outputs
        real :: det
        ! Local variables
        real :: adx, ady, bdx, bdy, cdx, cdy
        real :: abdet, bcdet, cadet, alift, blift, clift

        adx = a(1) - p(1)
        ady = a(2) - p(2)
        bdx = b(1) - p(1)
        bdy = b(2) - p(2)
        cdx = c(1) - p(1)
        cdy = c(2) - p(2)

        abdet = adx*bdy - bdx*ady
        bcdet = bdx*cdy - cdx*bdy
        cadet = cdx*ady - adx*cdy
        alift = adx*adx + ady*ady
        blift = bdx*bdx + bdy*bdy
        clift = cdx*cdx + cdy*cdy
        det = alift*bcdet + blift*cadet + clift*abdet
    end function incircle_real

    !> Calculates the signed in-circle determinant for 32-bit
    !! coordinates.
    !!
    !! The calculation uses double-precision intermediates. A result
    !! outside the 32-bit range is saturated to the corresponding
    !! integer limit. The result retains the 32-bit input kind and
    !! preserves the predicate sign.
    pure function incircle_int32(a, b, c, p) result(det)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: a(2), b(2), c(2), p(2)
        ! Outputs
        integer(c_int32_t) :: det

        det = saturate_int32( &
              incircle_double(real(a, kind=c_double), real(b, kind=c_double), &
                              real(c, kind=c_double), real(p, kind=c_double)))
    end function incircle_int32

    !> Calculates the sign of the in-circle determinant for 32-bit
    !! coordinates.
    pure function incircle_pos_int32(a, b, c, p) result(pos)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: a(2), b(2), c(2), p(2)
        ! Outputs
        logical(kind=1) :: pos

        pos = incircle_double( &
              real(a, kind=c_double), real(b, kind=c_double), &
              real(c, kind=c_double), real(p, kind=c_double)) > 0
    end function incircle_pos_int32

    !> Calculates the signed in-circle determinant for 64-bit
    !! coordinates.
    !!
    !! The calculation uses double-precision intermediates. A result
    !! outside the 64-bit range is saturated to the corresponding
    !! integer limit. The result retains the 64-bit input kind and
    !! preserves the predicate sign.
    pure function incircle_int64(a, b, c, p) result(det)
        implicit none(type, external)
        ! Arguments
        !f2py integer(kind=8), intent(in) :: a, b, c, p
        integer(c_int64_t), intent(in) :: a(2), b(2), c(2), p(2)
        ! Outputs
        !f2py integer(kind=8) :: det
        integer(c_int64_t) :: det

        det = saturate_int64( &
              incircle_double(real(a, kind=c_double), real(b, kind=c_double), &
                              real(c, kind=c_double), real(p, kind=c_double)))
    end function incircle_int64

    !> Calculates the signed in-circle determinant.
    pure function incircle_double(a, b, c, p) result(det)
        ! Double-precision implementation shared by the integer
        ! entry points.
        implicit none(type, external)
        ! Arguments
        real(c_double), intent(in) :: a(2), b(2), c(2), p(2)
        ! Outputs
        real(c_double) :: det
        ! Local variables
        real(c_double) :: adx, ady, bdx, bdy, cdx, cdy
        real(c_double) :: abdet, bcdet, cadet, alift, blift, clift

        adx = a(1) - p(1)
        ady = a(2) - p(2)
        bdx = b(1) - p(1)
        bdy = b(2) - p(2)
        cdx = c(1) - p(1)
        cdy = c(2) - p(2)

        abdet = adx*bdy - bdx*ady
        bcdet = bdx*cdy - cdx*bdy
        cadet = cdx*ady - adx*cdy
        alift = adx*adx + ady*ady
        blift = bdx*bdx + bdy*bdy
        clift = cdx*cdx + cdy*cdy
        det = alift*bcdet + blift*cadet + clift*abdet
    end function incircle_double

    pure function on_segment(a, b, p) result(on_flag)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: a(2), b(2), p(2)
        ! Outputs
        logical :: on_flag

        if (orient_real(a, b, p) /= 0) then
            on_flag = .false.
            return
        end if

        on_flag = p(1) >= min(a(1), b(1)) .and. &
                  p(1) <= max(a(1), b(1)) .and. &
                  p(2) >= min(a(2), b(2)) .and. &
                  p(2) <= max(a(2), b(2))
    end function on_segment

    elemental function xcross_orient_real(o_abc, o_abd, o_cda, o_cdb) &
        result(flag)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: o_abc, o_abd, o_cda, o_cdb
        logical(kind=1) :: flag

        flag = ((o_abc /= 0) .and. (o_abd /= 0) .and. &
                (o_cda /= 0) .and. (o_cdb /= 0) .and. &
                ((o_abc > 0) .neqv. (o_abd > 0)) .and. &
                ((o_cda > 0) .neqv. (o_cdb > 0)))
    end function xcross_orient_real

    elemental function xcross_orient_int32(o_abc, o_abd, o_cda, o_cdb) &
        result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: o_abc, o_abd, o_cda, o_cdb
        logical(kind=1) :: flag

        flag = ((o_abc /= 0) .and. (o_abd /= 0) .and. &
                (o_cda /= 0) .and. (o_cdb /= 0) .and. &
                ((o_abc > 0) .neqv. (o_abd > 0)) .and. &
                ((o_cda > 0) .neqv. (o_cdb > 0)))
    end function xcross_orient_int32

    elemental function xcross_orient_int64(o_abc, o_abd, o_cda, o_cdb) &
        result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int64_t), intent(in) :: o_abc, o_abd, o_cda, o_cdb
        logical(kind=1) :: flag

        flag = ((o_abc /= 0) .and. (o_abd /= 0) .and. &
                (o_cda /= 0) .and. (o_cdb /= 0) .and. &
                ((o_abc > 0) .neqv. (o_abd > 0)) .and. &
                ((o_cda > 0) .neqv. (o_cdb > 0)))
    end function xcross_orient_int64

    pure function xcross_real(a, b, c, d) result(flag)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: a(2), b(2), c(2), d(2)
        logical(kind=1) :: flag
        real :: o_abc, o_abd, o_cda, o_cdb

        o_abc = orient(a, b, c)
        o_abd = orient(a, b, d)
        o_cda = orient(c, d, a)
        o_cdb = orient(c, d, b)
        flag = xcross_orient(o_abc, o_abd, o_cda, o_cdb)
    end function xcross_real

    pure function xcross_int32(a, b, c, d) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: a(2), b(2), c(2), d(2)
        logical(kind=1) :: flag
        integer(c_int32_t) :: o_abc, o_abd, o_cda, o_cdb

        o_abc = orient(a, b, c)
        o_abd = orient(a, b, d)
        o_cda = orient(c, d, a)
        o_cdb = orient(c, d, b)
        flag = xcross_orient(o_abc, o_abd, o_cda, o_cdb)
    end function xcross_int32

    pure function xcross_int64(a, b, c, d) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int64_t), intent(in) :: a(2), b(2), c(2), d(2)
        logical(kind=1) :: flag
        integer(c_int64_t) :: o_abc, o_abd, o_cda, o_cdb

        o_abc = orient(a, b, c)
        o_abd = orient(a, b, d)
        o_cda = orient(c, d, a)
        o_cdb = orient(c, d, b)
        flag = xcross_orient(o_abc, o_abd, o_cda, o_cdb)
    end function xcross_int64

    !> Classifies the intersection of 2 closed 2D line segments.
    !!
    !! Flags:
    !! - -1 : disjoint
    !! -  0 : endpoint-to-endpoint touch
    !! -  1 : interior-interior crossing (X)
    !! -  2 : collinear overlap, not identical
    !! -  3 : identical segment
    !! -  4 : endpoint-on-interior (T-junction)
    !! -  5 : degenerate segment (some line is actually a point)
    pure function lines_intersect(l1a, l1b, l2a, l2b) result(flag)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: l1a(2), l1b(2), l2a(2), l2b(2)
        ! Outputs
        integer(c_int8_t) :: flag
        ! Local variables
        logical(kind=1) :: eq_l1al2a, eq_l1al2b, eq_l1bl2a, eq_l1bl2b
        real :: o1, o2, o3, o4
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

        ! Interior-interior crossing
        o1 = orient_real(l1a, l1b, l2a)
        o2 = orient_real(l1a, l1b, l2b)
        o3 = orient_real(l2a, l2b, l1a)
        o4 = orient_real(l2a, l2b, l1b)
        if (xcross_orient(o1, o2, o3, o4)) then
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
    end function lines_intersect

end module intersections
