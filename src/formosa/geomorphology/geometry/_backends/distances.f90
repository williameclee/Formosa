!> Compute distance metrics between points using the FORTRAN backend.
!!
!! This internal module provides geometric primitives used by other
!! FORTRAN routines and is not intended to be used directly.
!!
!! Last modified: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
module distances
    use iso_c_binding, only: c_int8_t, c_int32_t
    implicit none(type, external)
    interface l1dist_xy
        module procedure l1dist_xy_int
        module procedure l1dist_xy_real
    end interface l1dist_xy
    interface l2dist2_xy
        module procedure l2dist2_xy_int32
        module procedure l2dist2_xy_real
    end interface l2dist2_xy
    interface l2dist_xy
        module procedure l2dist_xy_int32
        module procedure l2dist_xy_real
    end interface l2dist_xy
contains
    pure function l1dist_xy_int(x1, y1, x2, y2) result(dist)
        !! Calculates the L1 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: x1, y1, x2, y2
        integer :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_int

    pure function l1dist_xy_real(x1, y1, x2, y2) result(dist)
        !! Calculates the L1 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_real

    pure function l2dist2_xy_int32(x1, y1, x2, y2) result(dist2)
        !! Calculates the L2 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: x1, y1, x2, y2
        integer(c_int32_t) :: dist2
        dist2 = (x1 - x2)**2 + (y1 - y2)**2
    end function l2dist2_xy_int32

    pure function l2dist2_xy_real(x1, y1, x2, y2) result(dist2)
        !! Calculates the L2 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist2
        dist2 = (x1 - x2)**2 + (y1 - y2)**2
    end function l2dist2_xy_real

    pure function l2dist_xy_int32(x1, y1, x2, y2) result(dist)
        !! Calculates the L2 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: x1, y1, x2, y2
        real :: dist
        ! 'sqrt' is unsed instead of 'hypot' because coordinates are bounded
        ! and do not present underflow/overflow risk here.
        dist = sqrt(real(l2dist2_xy(x1, y1, x2, y2)))
    end function l2dist_xy_int32

    pure function l2dist_xy_real(x1, y1, x2, y2) result(dist)
        !! Calculates the L2 distance between two 2D points.
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        ! 'sqrt' is unsed instead of 'hypot' because coordinates are bounded
        ! and do not present underflow/overflow risk here.
        dist = sqrt(l2dist2_xy(x1, y1, x2, y2))
    end function l2dist_xy_real

    pure function pt2linedist2_xy(x1, y1, x2, y2, x3, y3) result(dist2)
        ! Calculates the squared distance of a point to a line segment defined by two points on a 2D plane.
        implicit none(type, external)
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
        implicit none(type, external)
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
end module distances
