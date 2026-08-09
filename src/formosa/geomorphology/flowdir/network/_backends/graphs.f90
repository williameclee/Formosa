!!!
! Last modified
!   2026-07-02, En-Chi Lee (williameclee@gmail.com)
!     - Added function 'construct_flowgraph'
!   2026-07-09, En-Chi Lee (williameclee@gmail.com)
!     - Added overflow check in 'construct_flowgraph'
!     - Implemented 'simplify_flowgraph' function
!   2026-07-14, En-Chi Lee (williameclee@gmail.com)
!     - Splitted 'flowdir_f' into submodules
!   2026-08-03, En-Chi Lee (williameclee@gmail.com)
!     - Explicitly handled Python uint8 -> int8 FORTRAN conversion/
!       interpretation in 'fill_offset_lookup'
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Added allocation error monitoring and moved error handling
!       to Python
!   2026-08-05, En-Chi Lee (williameclee@gmail.com)
!     - Switched to 'iso_c_binding'
!!!

module flowdir_graphs
    use iso_c_binding, only: c_int8_t, c_int16_t
    use utils, only: fill_offset_lookup, find_noflow_code, mask2ij
    use geometry, only: pt2linedist2_xy
    use intersections, only: lines_intersect_v2
    implicit none(type, external)
contains
    subroutine construct_flowgraph( &
        dirs, valids, orders, seeds, indegs, nrows, ncols, &
        offsets, codes, noffsets, preserve_junction, ncells, &
        narcs, nvertices, arc_orders, vertex_ijs, arc_endpts, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for cells that should not be processed, including those with low order)
        integer(c_int16_t), intent(in) :: orders(nrows, ncols)
            !! Grid of Strahler stream order values for each cell
        logical(kind=1), intent(in) :: seeds(nrows, ncols)
            !! Mask to identify initial seed cells for the algorithm (valid cells with zero indegree)
        integer(c_int8_t), intent(in) :: indegs(nrows, ncols)
            !! Indegree of the cell
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        logical, intent(in) :: preserve_junction
            !! Whether to stop an arc when another arc joins it
        integer, intent(in) :: ncells
            !! Number of valid cells
        ! Outputs
        integer, intent(out) :: narcs, nvertices
            !! How many arcs and vertices there are
        integer(c_int16_t), intent(out) :: arc_orders(ncells)
            !! Order of each arc
            !! Note only the first 'narcs' elements contain the actual data
        integer, intent(out) :: vertex_ijs(2, 2*ncells)
            !! Ordered (i, j) indices of cells that each arc contains
            !! Note only the first 'nvertices' columns contain the actual data
        integer, intent(out) :: arc_endpts(2, ncells)
            !! Where each arc starts and ends in the 'vertex_ijs' array
            !! Note only the first 'narcs' columns contain the actual data
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 2: Internal workspace allocation failed
            !!   - 3: Vertex output buffer capacity was exceeded
        ! Local variables
        integer(c_int8_t) :: noflow_code
            !! Code corresponding to noflow direction, used to identify sink cells
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer, allocatable :: seed_ijs(:, :)
            !! Buffer for storing (i, j) indices of seed cells
        integer(c_int16_t) :: order
            !! Order of the current arc
        integer :: nseeds, iseed
            !! Number of seeds and index for iterating through seeds
        integer :: iarc, ivertex
            !! index for iterating through arcs and vertices
        integer :: si, sj, ci, cj, ni, nj
            !! Rows/columns for seed, current, and neighbour cells
        logical :: ds_is_valid, is_end_vertex
            !! Flag of whether the downstream neighbour is a valid cell, and whether we have arrived at the end of the arc
        logical(kind=1), allocatable :: seens(:, :)
            !! Mask to identify which cells have already been seen
        integer :: alloc_stat
            !! Allocation status code

        err_code = 0
        narcs = 0
        nvertices = 0

        ! Create lookup tables for offsets
        allocate (offset_lookup(0:255, 2), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        ! Find index of seeds
        allocate (seed_ijs(2, ncells), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            return
        end if
        call mask2ij(seeds, seed_ijs, ncells, nseeds, err_code)
        if (err_code /= 0) return

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes)

        allocate (seens(nrows, ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            deallocate (seed_ijs)
            return
        end if
        seens = .false.
        iseed = 1
        iarc = 1
        ivertex = 1

        do while (iseed <= nseeds)
            si = seed_ijs(1, iseed)
            sj = seed_ijs(2, iseed)
            iseed = iseed + 1
            seens(si, sj) = .true.

            ! Skip isolated point
            if (dirs(si, sj) == noflow_code) cycle

            ! Initialise the arc
            order = orders(si, sj)
            arc_orders(iarc) = order
            arc_endpts(1, iarc) = ivertex
            vertex_ijs(:, ivertex) = [si, sj]
            ivertex = ivertex + 1
            ci = si
            cj = sj

            do while (.true.)
                ! First check the downstream cell
                ni = ci + offset_lookup(iand(int(dirs(ci, cj)), 255), 1)
                nj = cj + offset_lookup(iand(int(dirs(ci, cj)), 255), 2)

                ds_is_valid = .true.
                if (ci == ni .and. cj == nj) then ! Self-loop
                    ds_is_valid = .false.
                else if (ni <= 0 .or. ni > nrows .or. nj <= 0 .or. nj > ncols) then ! OOB
                    ds_is_valid = .false.
                else if (.not. valids(ni, nj)) then
                    ds_is_valid = .false.
                end if

                if (.not. ds_is_valid) then
                    is_end_vertex = .true.
                else
                    is_end_vertex = orders(ni, nj) /= order
                    if (preserve_junction) is_end_vertex = is_end_vertex .or. (indegs(ni, nj) >= 2)
                end if

                if (is_end_vertex) then
                    if (.not. ds_is_valid) then
                        if (arc_endpts(1, iarc) == ivertex - 1) then
                            ! Single-length arc, roll back arc and vertex registration
                            ivertex = ivertex - 1
                            iarc = iarc - 1
                            exit
                        else
                            arc_endpts(2, iarc) = ivertex - 1
                            exit
                        end if
                    end if
                    if (ivertex > size(vertex_ijs, 2)) then
                        err_code = 3
                        exit
                    end if
                    vertex_ijs(:, ivertex) = [ni, nj]
                    arc_endpts(2, iarc) = ivertex
                    ivertex = ivertex + 1
                    if (ds_is_valid .and. (.not. seens(ni, nj))) then
                        seens(ni, nj) = .true.
                        nseeds = nseeds + 1
                        seed_ijs(:, nseeds) = [ni, nj]
                    end if
                    exit
                end if

                seens(ni, nj) = .true.
                if (ivertex > size(vertex_ijs, 2)) then
                    err_code = 3
                    exit
                end if
                vertex_ijs(:, ivertex) = [ni, nj]
                ivertex = ivertex + 1
                ci = ni
                cj = nj
            end do

            if (err_code /= 0) exit

            iarc = iarc + 1
        end do

        deallocate (offset_lookup)
        deallocate (seens)
        deallocate (seed_ijs)

        narcs = iarc - 1
        nvertices = ivertex - 1
    end subroutine construct_flowgraph

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
end module flowdir_graphs
