!!!
! Last modified
!   2026-07-02, En-Chi Lee (williameclee@gmail.com)
!     - Added function 'construct_flowgraph'
!   2026-07-09, En-Chi Lee (williameclee@gmail.com)
!     - Added overflow check in 'construct_flowgraph'
!     - Implemented 'simplify_flowgraph' function
!   2026-07-12, En-Chi Lee (williameclee@gmail.com)
!     - Implemented 'locate_invalid_graph_topology' function
!   2026-07-14, En-Chi Lee (williameclee@gmail.com)
!     - Splitted 'flowdir_f' into submodules
!   2026-07-29, En-Chi Lee (williameclee@gmail.com)
!     - Made topology intersection scans count all violations past output capacity
!!!

module flowdir_graphs
    use omp_lib
    use utils
    use distances
    implicit none
    private :: argsort_arcs, record_topology_intersection
contains
    subroutine construct_flowgraph( &
        dirs, valids, orders, seeds, indegs, nrows, ncols, &
        offsets, codes, noffsets, preserve_junction, ncells, &
        narcs, nvertices, arc_orders, vertex_ijs, arc_endpts)
        implicit none
        ! Arguments
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer*1, intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        logical*1, intent(in) :: valids(nrows, ncols)
            !! Validity mask (true for valid cells, false for cells that should not be processed, including those with low order)
        integer*2, intent(in) :: orders(nrows, ncols)
            !! Grid of Strahler stream order values for each cell
        logical*1, intent(in) :: seeds(nrows, ncols)
            !! Mask to identify initial seed cells for the algorithm (valid cells with zero indegree)
        integer*1, intent(in) :: indegs(nrows, ncols)
            !! Indegree of the cell
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer*1, intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        logical, intent(in) :: preserve_junction
            !! Whether to stop an arc when another arc joins it
        integer, intent(in) :: ncells
            !! Number of valid cells
        ! Outputs
        integer, intent(out) :: narcs, nvertices
            !! How many arcs and vertices there are
        integer*2, intent(out) :: arc_orders(ncells)
            !! Order of each arc
            !! Note only the first 'narcs' elements contain the actual data
        integer, intent(out) :: vertex_ijs(2, 2*ncells)
            !! Ordered (i, j) indices of cells that each arc contains
            !! Note only the first 'nvertices' columns contain the actual data
        integer, intent(out) :: arc_endpts(2, ncells)
            !! Where each arc starts and ends in the 'vertex_ijs' array
            !! Note only the first 'narcs' columns contain the actual data
        ! Local variables
        integer*1 :: noflow_code
            !! Code corresponding to noflow direction, used to identify sink cells
        integer, allocatable :: offset_lookup(:, :)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        integer, allocatable :: seed_ijs(:, :)
            !! Buffer for storing (i, j) indices of seed cells
        integer*2 :: order
            !! Order of the current arc
        integer :: nseeds, iseed
            !! Number of seeds and index for iterating through seeds
        integer :: iarc, ivertex
            !! index for iterating through arcs and vertices
        integer :: si, sj, ci, cj, ni, nj
            !! Rows/columns for seed, current, and neighbour cells
        logical :: ds_is_valid, is_end_vertex
            !! Flag of whether the downstream neighbour is a valid cell, and whether we have arrived at the end of the arc
        logical*1, allocatable :: seens(:, :)
            !! Mask to identify which cells have already been seen

        ! Create lookup tables for offsets
        allocate (offset_lookup(0:255, 2))
        offset_lookup = fill_offset_lookup(offsets, codes, noffsets)

        ! Find index of seeds
        allocate (seed_ijs(2, ncells))
        call mask2ij(seeds, nrows, ncols, seed_ijs, ncells, nseeds)

        ! Find noflow code
        noflow_code = find_noflow_code(offsets, codes, noffsets)

        allocate (seens(nrows, ncols))
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
                ni = ci + offset_lookup(dirs(ci, cj), 1)
                nj = cj + offset_lookup(dirs(ci, cj), 2)

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
                        print *, "[CONSTRUCT_FLOWGRAPH] Error: vertex buffer overflow "// &
                            "(size:", ivertex, ", allocated:", size(vertex_ijs, 2), ")"
                        stop
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
                    print *, "[CONSTRUCT_FLOWGRAPH] Error: vertex buffer overflow "// &
                        "(size:", ivertex, ", allocated:", size(vertex_ijs, 2), ")"
                    stop
                end if
                vertex_ijs(:, ivertex) = [ni, nj]
                ivertex = ivertex + 1
                ci = ni
                cj = nj
            end do

            iarc = iarc + 1
        end do

        deallocate (offset_lookup)
        deallocate (seens)
        deallocate (seed_ijs)

        narcs = iarc - 1
        nvertices = ivertex - 1
    end subroutine construct_flowgraph

    recursive subroutine simplify_arc_rdp( &
        xys, keeps, istart, iend, tol)
        ! Simplify a single arc segment recursively using the Ramer-Douglas-Peucker (RDP) algorithm.
        implicit none
        ! Arguments
        real, intent(in) :: xys(:, :)
            !! x and y coordinates of each vertex
        integer, intent(in) :: istart, iend
            !! Where the segment starts and ends in the 'xys' array
        real, intent(in) :: tol
            !! Tolerence threshold
        ! Outputs
        logical*1, intent(inout) :: keeps(:)
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

    subroutine simplify_flowgraph( &
        vertex_xys, arc_endpts, vertex_keeps, nvertices, narcs, tol)
        ! Simplify all arcs in a flow graph using the Ramer-Douglas-Peucker (RDP) algorithm.
        implicit none
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
        logical*1, intent(out) :: vertex_keeps(nvertices)
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

    pure function argsort_arcs(bboxes) result(indices)
        ! Helper function for 'locate_invalid_graph_topology' to sort the arcs by the left edge of their bounding box.
        implicit none
        ! Arguments
        real, intent(in) :: bboxes(:, :)
        ! Outputs
        integer :: indices(size(bboxes, 2))
        ! Local variables
        integer :: i, j, h, index

        do i = lbound(bboxes, 2), ubound(bboxes, 2)
            indices(i) = i
        end do

        ! Shell sort idx by the left edge
        h = ubound(bboxes, 2)/2
        do while (h > 0)
            do i = h + 1, ubound(bboxes, 2)
                index = indices(i)
                j = i
                do while (j > h)
                    if (bboxes(1, indices(j - h)) > bboxes(1, index)) then
                        indices(j) = indices(j - h)
                        j = j - h
                    else
                        exit
                    end if
                end do
                indices(j) = index
            end do
            h = h/2
        end do
    end function argsort_arcs

    subroutine record_topology_intersection(record, intxs, nintxs)
        implicit none
        integer, intent(in) :: record(5)
        integer, intent(inout) :: intxs(:, :)
        integer, intent(inout) :: nintxs

        nintxs = nintxs + 1
        if (nintxs <= size(intxs, 2)) intxs(:, nintxs) = record
    end subroutine record_topology_intersection

    subroutine scan_invalid_graph_topology( &
        vertex_ijs, arc_endpts, capacity, intxs, nintxs, err_code)
        !! Scans all candidate segment pairs and returns the total violation count.
        !! Only the first 'capacity' violations are stored in 'intxs'.
        implicit none
        ! Arguments
        real, intent(in) :: vertex_ijs(:, :)
        integer, intent(in) :: arc_endpts(:, :)
        integer, intent(in) :: capacity
        ! Outputs
        integer, intent(out) :: intxs(5, capacity)
        integer, intent(out) :: nintxs
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: Input dimensions are incorrect, or input capacity is invalid
            !!   - 2: Memory allocation failed
        ! Local variables
        integer :: narcs
        integer :: i, j, iarc, jarc, iseg, jseg
        integer :: intx_flag, alloc_stat
        real, allocatable :: arc_bboxes(:, :)
        integer, allocatable :: idx(:)

        nintxs = 0
        err_code = 0

        if (size(arc_endpts, 1) /= 2) then
            err_code = 1
            return
        else if (size(vertex_ijs, 1) /= 2) then
            err_code = 1
            return
        else if (capacity < 1) then
            err_code = 1
            return
        end if
        narcs = size(arc_endpts, 2)

        if (narcs == 0) return

        ! Construct the bounding boxes for each arc
        allocate (arc_bboxes(4, narcs), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            return
        end if
        do iarc = 1, narcs
            arc_bboxes(1, iarc) = minval(vertex_ijs(1, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(2, iarc) = minval(vertex_ijs(2, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(3, iarc) = maxval(vertex_ijs(1, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(4, iarc) = maxval(vertex_ijs(2, arc_endpts(1, iarc):arc_endpts(2, iarc)))
        end do

        ! Check arcs against themselves first
        do iarc = 1, narcs
            if (arc_endpts(2, iarc) - arc_endpts(1, iarc) == 1) cycle ! Skip if arc is just a single segment
            do iseg = arc_endpts(1, iarc), arc_endpts(2, iarc) - 1
            do jseg = iseg + 1, arc_endpts(2, iarc) - 1
                intx_flag = lines_intersect_v2( &
                            vertex_ijs(:, iseg), vertex_ijs(:, iseg + 1), &
                            vertex_ijs(:, jseg), vertex_ijs(:, jseg + 1))
                if (intx_flag > 0) then
                    call record_topology_intersection( &
                        [iarc, iarc, iseg, jseg, intx_flag], intxs, nintxs)
                end if
            end do
            end do
        end do

        allocate (idx(narcs), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (arc_bboxes)
            return
        end if
        idx = argsort_arcs(arc_bboxes)

        ! Check every arc against each other
        do i = 1, narcs
            iarc = idx(i)
            do j = i + 1, narcs
                jarc = idx(j)

                ! Skip if min x of right arc is greater than max x of left arc
                if (arc_bboxes(1, jarc) > arc_bboxes(3, iarc)) exit

                ! Inline fast overlap check (no min/max calls)
                if (arc_bboxes(1, iarc) > arc_bboxes(3, jarc) .or. &
                    arc_bboxes(3, iarc) < arc_bboxes(1, jarc) .or. &
                    arc_bboxes(2, iarc) > arc_bboxes(4, jarc) .or. &
                    arc_bboxes(4, iarc) < arc_bboxes(2, jarc)) cycle

                do iseg = arc_endpts(1, iarc), arc_endpts(2, iarc) - 1
                do jseg = arc_endpts(1, jarc), arc_endpts(2, jarc) - 1
                    intx_flag = lines_intersect_v2( &
                                vertex_ijs(:, iseg), vertex_ijs(:, iseg + 1), &
                                vertex_ijs(:, jseg), vertex_ijs(:, jseg + 1))
                    if (intx_flag > 0) then
                        ! Sort by arc ID
                        if (iarc < jarc) then
                            call record_topology_intersection( &
                                [iarc, jarc, iseg, jseg, intx_flag], intxs, nintxs)
                        else
                            call record_topology_intersection( &
                                [jarc, iarc, jseg, iseg, intx_flag], intxs, nintxs)
                        end if
                    end if
                end do
                end do
            end do
        end do

        deallocate (idx)
        deallocate (arc_bboxes)
    end subroutine scan_invalid_graph_topology
end module flowdir_graphs
