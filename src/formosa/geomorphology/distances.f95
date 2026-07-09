module distances
    implicit none
    interface l1dist_xy
        module procedure l1dist_xy_int
        module procedure l1dist_xy_real
    end interface l1dist_xy
contains
    function l1dist_xy_int(x1, y1, x2, y2) result(dist)
        implicit none
        ! Arguments
        integer, intent(in) :: x1, y1, x2, y2
        integer :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_int

    function l1dist_xy_real(x1, y1, x2, y2) result(dist)
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        dist = abs(x1 - x2) + abs(y1 - y2)
    end function l1dist_xy_real

    function l2dist_xy(x1, y1, x2, y2) result(dist)
        implicit none
        ! Arguments
        real, intent(in) :: x1, y1, x2, y2
        real :: dist
        ! 'sqrt' is unsed instead of 'hypot' because coordinates are bounded
        ! and do not present underflow/overflow risk here.
        dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    end function l2dist_xy
end module distances
