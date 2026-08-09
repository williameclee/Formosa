!!!
! Last modified
!   2026-07-09, En-Chi Lee (williameclee@gmail.com)
!     - Implemented 'simplify_flowgraph' function
!   2026-07-14, En-Chi Lee (williameclee@gmail.com)
!     - Splitted 'flowdir_f' into submodules
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Added allocation error monitoring and moved error handling
!       to Python
!!!

module network_simplification
    use geometry, only: pt2linedist2_xy
    implicit none(type, external)
contains
    pure recursive subroutine simplify_arc_rdp( &
        xys, keeps, istart, iend, tol)
        ! Simplify a single arc segment recursively using the Ramer-Douglas-Peucker (RDP) algorithm.
        implicit none(type, external)
        ! Arguments
        real, intent(in), contiguous :: xys(:, :)
            !! x and y coordinates of each vertex
        integer, intent(in) :: istart, iend
            !! Where the segment starts and ends in the 'xys' array
        real, intent(in) :: tol
            !! Tolerence threshold
        ! Outputs
        logical(kind=1), intent(inout), contiguous :: keeps(:)
            !! Boolean mask indicating which vertices should be kept
        ! Local variables
        integer :: i
            !! Index for iterating through vertices
        real :: err2, max_err2
            !! Squared error at individual points and the maximum squared error
        integer :: i_max_err2
            !! Index of the point with the maximum error

        ! Initialisation
        keeps(istart) = .true.
        keeps(iend) = .true.

        if ((iend - istart) <= 1) return ! Noting to do if <= 2 points

        ! Find the point with the maximum perpendicular distance to the segment line
        max_err2 = 0.
        i_max_err2 = istart
        do i = istart + 1, iend - 1
            err2 = pt2linedist2_xy(xys(1, istart), xys(2, istart), xys(1, iend), xys(2, iend), xys(1, i), xys(2, i))
            if (err2 <= max_err2) cycle
            max_err2 = err2
            i_max_err2 = i
        end do

        ! If max error is within the tolerance threshold, simplify (keep only endpoints)
        if (max_err2 <= tol**2) return
        ! Otherwise, keep the point with maximum error and recursively simplify the two sub-segments
        keeps(i_max_err2) = .true.
        call simplify_arc_rdp(xys, keeps, istart, i_max_err2, tol)
        call simplify_arc_rdp(xys, keeps, i_max_err2, iend, tol)
    end subroutine simplify_arc_rdp

    pure subroutine simplify_flowgraph( &
        vertex_xys, arc_endpts, vertex_keeps, nvertices, narcs, tol)
        ! Simplify all arcs in a flow graph using the Ramer-Douglas-Peucker (RDP) algorithm.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvertices, narcs
            !! Number of vertices and arcs
        real, intent(in) :: vertex_xys(2, nvertices)
            !! x and y coordinates of each vertex
        integer, intent(in) :: arc_endpts(2, narcs)
            !! Where each arc starts and ends in the 'vertex_xys' array
        real, intent(in) :: tol
            !! Tolerence threshold
        ! Outputs
        logical(kind=1), intent(out) :: vertex_keeps(nvertices)
            !! Boolean mask indicating which vertices should be kept across all arcs
        ! Local variables
        integer :: iarc
            !! Index for iterating through arcs

        ! Initialisation
        vertex_keeps = .false.
        ! Simplify each arc independently
        do iarc = 1, narcs
            call simplify_arc_rdp(vertex_xys, vertex_keeps, arc_endpts(1, iarc), arc_endpts(2, iarc), tol)
        end do
    end subroutine simplify_flowgraph
end module network_simplification
