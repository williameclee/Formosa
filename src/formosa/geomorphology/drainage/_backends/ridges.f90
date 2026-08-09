!!!
! Last modified
!   2026-07-01, En-Chi Lee (williameclee@gmail.com)
!     - Optimised confluence lookup algorithm
!   2026-08-04, En-Chi Lee (williameclee@gmail.com)
!     - Added allocation error monitoring and moved error handling
!       to Python
!   2026-08-05, En-Chi Lee (williameclee@gmail.com)
!     - Overhauled algorithm for 'compute_max_branch_dist'
!     - Switched to 'iso_c_binding'
!!!

module drainage_ridges
    use iso_c_binding, only: c_int8_t
    use utils, only: fill_offset_lookup, array2d_oob, ij2id_checked
    use geometry, only: l2dist_xy
    implicit none(type, external)
    private :: resolve_flowtree_links, build_flowtree_topology, &
               propagate_flowtree_metadata, build_flowtree_metadata, &
               find_flowtree_confluence
contains
    subroutine resolve_flowtree_links( &
        dirs, valids, offset_lookup, nrows, ncols, ds_ids, indegs)
        !! Resolves each valid cell's immediate downstream ID and
        !! simultaneously count the upstream children of every
        !! destination cell.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction code for every raster cell
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Row/column offset indexed by the unsigned direction code.
        integer, intent(out) :: ds_ids(nrows*ncols)
            !! Immediate downstream cell ID, or zero at a sink.
        integer(c_int8_t), intent(out) :: indegs(nrows*ncols)
            !! Number of valid upstream children targeting each cell.
        integer :: code
            !! Unsigned integer representation of the current direction code.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream destination.
        integer :: ci, cj, ni, nj
            !! Row/column coordinates of the current and downstream cells.

        ds_ids = 0
        indegs = 0

        ! Resolve one downstream edge per valid source cell. Coordinates are
        ! checked before linear encoding so an invalid pair cannot wrap into a
        ! different, apparently legitimate cell ID.
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(ci, cj, ni, nj, code, cid, dsid) &
        !$omp COLLAPSE(2) SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                cid = ij2id_checked(ci, cj, nrows, ncols)
                code = iand(int(dirs(ci, cj)), 255)
                if (offset_lookup(code, 1) == -99 .and. offset_lookup(code, 2) == -99) cycle
                ni = ci + offset_lookup(code, 1)
                nj = cj + offset_lookup(code, 2)
                if (array2d_oob(ni, nj, nrows, ncols)) cycle
                if (.not. valids(ni, nj)) cycle
                dsid = ij2id_checked(ni, nj, nrows, ncols)
                if (dsid == 0 .or. dsid == cid) cycle
                ds_ids(cid) = dsid
                !$omp ATOMIC UPDATE
                indegs(dsid) = indegs(dsid) + int(1, kind=c_int8_t)
                !$omp END ATOMIC
            end do
        end do
        !$omp END PARALLEL DO
    end subroutine resolve_flowtree_links

    subroutine build_flowtree_topology( &
        valids, ds_ids, indegs, nrows, ncols, &
        topo_order, topo_cnt, lvl_ends, nlvls, err_code)
        !! Build source-to-sink Kahn frontiers and reject directed cycles.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells belonging to the flow tree; false for no-data.
        integer, intent(in) :: ds_ids(nrows*ncols)
            !! Immediate downstream ID for every cell; zero at sinks.
        integer(c_int8_t), intent(inout) :: indegs(nrows*ncols)
            !! Remaining unprocessed upstream-child count for Kahn traversal.
        integer, intent(out) :: topo_order(nrows*ncols)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        integer, intent(out) :: topo_cnt
            !! Number of valid entries written to topo_order.
        integer, allocatable, intent(inout) :: lvl_ends(:)
            !! Inclusive topo_order end position of every Kahn frontier.
        integer, intent(out) :: nlvls
            !! Number of dependency frontiers recorded in lvl_ends.
        integer, intent(out) :: err_code
            !! Zero on success, one for a cycle, or two for allocation failure.
        ! Local variables
        integer, allocatable :: grown_lvl_ends(:)
            !! Temporary buffer used when geometrically growing lvl_ends.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream destination.
        integer :: ci, cj
            !! Row/column coordinates used while identifying valid sources.
        integer :: sorder
            !! Current read position of topo_order.
        integer :: lvl_start, lvl_end, next_lvl_end
            !! Inclusive bounds of the active Kahn frontier and its appended end.
        integer :: nvalid
            !! Number of valid cells expected in a complete topological order.
        integer :: new_lvl_capacity, alloc_stat
            !! Requested level-buffer capacity and allocation status.

        err_code = 0
        topo_cnt = 0
        lvl_end = 0
        nvalid = 0

        ! Count number of valid cells and push 0-indegree cells into 'topo_order'
        do cid = 1, nrows*ncols
            ci = mod(cid - 1, nrows) + 1
            cj = (cid - 1)/nrows + 1
            if (.not. valids(ci, cj)) cycle
            nvalid = nvalid + 1
            if (indegs(cid) /= 0) cycle
            lvl_end = lvl_end + 1
            topo_order(lvl_end) = cid
        end do

        nlvls = 0
        lvl_start = 1
        lvl_end = lvl_end
        ! Go through cells level by level
        do while (lvl_start <= lvl_end)
            nlvls = nlvls + 1

            ! Reallocate lvl_ends if needed
            if (nlvls > size(lvl_ends)) then
                new_lvl_capacity = min(size(lvl_ends)*2, nrows*ncols)
                allocate (grown_lvl_ends(new_lvl_capacity), stat=alloc_stat)
                if (alloc_stat /= 0) then
                    err_code = 2
                    return
                end if
                grown_lvl_ends(1:nlvls - 1) = lvl_ends
                call move_alloc(grown_lvl_ends, lvl_ends)
            end if

            lvl_ends(nlvls) = lvl_end
            next_lvl_end = lvl_end

            ! Sweep through all the downstream cells of this frontier
            do sorder = lvl_start, lvl_end
                cid = topo_order(sorder)
                dsid = ds_ids(cid)
                if (dsid == 0) cycle
                indegs(dsid) = indegs(dsid) - int(1, kind=c_int8_t)
                if (indegs(dsid) /= 0) cycle
                next_lvl_end = next_lvl_end + 1
                topo_order(next_lvl_end) = dsid
            end do
            lvl_start = lvl_end + 1
            lvl_end = next_lvl_end
        end do
        topo_cnt = lvl_end
        if (topo_cnt /= nvalid) err_code = 1
    end subroutine build_flowtree_topology

    subroutine propagate_flowtree_metadata( &
        ds_ids, x, y, topo_order, lvl_ends, nlvls, nrows, ncols, &
        depths, sink_ids, sink_dists)
        !! Propagates depth, sink identity, and metric distance from sinks towards
        !! sources in reverse dependency-frontier order.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols, nlvls
            !! Raster dimensions and number of dependency frontiers.
        integer, intent(in) :: ds_ids(nrows*ncols)
            !! Immediate downstream ID for every cell; zero at sinks.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate metric edge lengths.
        integer, intent(in) :: topo_order(nrows*ncols), lvl_ends(:)
            !! Source-to-sink cell order and inclusive frontier boundaries.
        integer, intent(out) :: depths(nrows*ncols)
            !! Number of downstream edges from each cell to its sink.
        integer, intent(out) :: sink_ids(nrows*ncols)
            !! Linear ID of the sink reached by each valid cell.
        real, intent(out) :: sink_dists(nrows*ncols)
            !! Cumulative metric distance from each valid cell to its sink.
        integer, parameter :: min_parallel_lvl = 32768
            !! Minimum frontier width worth entering an OpenMP parallel region.
        integer :: ilvl, lvl_start, lvl_end, sorder
            !! Current frontier, its inclusive bounds, and traversal cursor.
        integer :: cid, dsid
            !! Linear ID of the current cell and its downstream parent.
        integer :: ci, cj, dsi, dsj
            !! Row/column coordinates of the current cell and its downstream parent.

        depths = 0
        sink_ids = 0
        sink_dists = 0.0
        do ilvl = nlvls, 1, -1
            lvl_end = lvl_ends(ilvl)
            if (ilvl == 1) then
                lvl_start = 1
            else
                lvl_start = lvl_ends(ilvl - 1) + 1
            end if

            ! Check if worth parallelising
            if (lvl_end - lvl_start + 1 >= min_parallel_lvl) then
                !$omp PARALLEL DO DEFAULT(SHARED) &
                !$omp PRIVATE(sorder, cid, dsid, ci, cj, dsi, dsj) SCHEDULE(STATIC)
                do sorder = lvl_start, lvl_end
                    ! Calculate distance to sink based on its downstream's distance to sink
                    cid = topo_order(sorder)
                    dsid = ds_ids(cid)
                    if (dsid == 0) then
                        sink_ids(cid) = cid
                        cycle
                    end if
                    depths(cid) = depths(dsid) + 1
                    sink_ids(cid) = sink_ids(dsid)
                    ci = mod(cid - 1, nrows) + 1
                    cj = (cid - 1)/nrows + 1
                    dsi = mod(dsid - 1, nrows) + 1
                    dsj = (dsid - 1)/nrows + 1
                    sink_dists(cid) = sink_dists(dsid) &
                                      + l2dist_xy(x(ci, cj), y(ci, cj), x(dsi, dsj), y(dsi, dsj))
                end do
                !$omp END PARALLEL DO
            else
                ! Same loop as above
                do sorder = lvl_start, lvl_end
                    cid = topo_order(sorder)
                    dsid = ds_ids(cid)
                    if (dsid == 0) then
                        sink_ids(cid) = cid
                        cycle
                    end if
                    depths(cid) = depths(dsid) + 1
                    sink_ids(cid) = sink_ids(dsid)
                    ci = mod(cid - 1, nrows) + 1
                    cj = (cid - 1)/nrows + 1
                    dsi = mod(dsid - 1, nrows) + 1
                    dsj = (dsid - 1)/nrows + 1
                    sink_dists(cid) = sink_dists(dsid) &
                                      + l2dist_xy(x(ci, cj), y(ci, cj), x(dsi, dsj), y(dsi, dsj))
                end do
            end if
        end do
    end subroutine propagate_flowtree_metadata

    subroutine build_flowtree_metadata( &
        dirs, valids, x, y, offset_lookup, nrows, ncols, &
        ds_ids, depths, sink_ids, sink_dists, topo_order, topo_cnt, err_code)
        !! Coordinate construction of reusable metadata for the downstream tree.
        !!
        !! The work is deliberately split into three independently testable
        !! phases:
        !!
        !!   resolve_flow_tree_links       -- downstream IDs and indegrees;
        !!   build_flow_tree_topology      -- Kahn frontiers and cycle detection;
        !!   propagate_flow_tree_metadata  -- depths, sinks, and distances.
        !!
        !! topo_order is returned because compute_max_branch_dist reuses it to
        !! construct lowest common ancestor (LCA) jump pointers before releasing
        !! the full-grid workspace.
        implicit none(type, external)
        integer, intent(in) :: nrows, ncols
            !! Number of raster rows and columns.
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow-direction code for every raster cell.
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! True for cells belonging to the flow tree; false for no-data.
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Map-space coordinates used to calculate metric edge lengths.
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Row/column offset indexed by the unsigned direction code.
        integer, intent(out) :: ds_ids(nrows*ncols)
            !! Immediate downstream parent ID, or zero when the cell is a sink.
        integer, intent(out) :: depths(nrows*ncols)
            !! Number of downstream edges from each cell to its sink.
        integer, intent(out) :: sink_ids(nrows*ncols)
            !! Linear ID of the sink reached by each valid cell.
        real, intent(out) :: sink_dists(nrows*ncols)
            !! Cumulative metric distance from each valid cell to its sink.
        integer, allocatable, intent(out) :: topo_order(:)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        integer, intent(out) :: topo_cnt
            !! Number of valid entries written to topo_order.
        integer, intent(out) :: err_code
            !! Zero on success, one for a cycle, or two for allocation failure.
        integer(c_int8_t), allocatable :: indegs(:)
            !! Mutable upstream-child counts consumed by Kahn traversal.
        integer, allocatable :: lvl_ends(:)
            !! Inclusive topo_order end position for each dependency frontier.
        integer :: nlvls
            !! Number of dependency frontiers recorded in lvl_ends.
        integer :: alloc_stat
            !! Status returned by workspace allocation statements.

        err_code = 0
        topo_cnt = 0
        allocate (indegs(nrows*ncols), topo_order(nrows*ncols), &
                  lvl_ends(max(1, min(nrows*ncols, 1024))), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            if (allocated(indegs)) deallocate (indegs)
            if (allocated(topo_order)) deallocate (topo_order)
            if (allocated(lvl_ends)) deallocate (lvl_ends)
            return
        end if

        call resolve_flowtree_links( &
            dirs, valids, offset_lookup, nrows, ncols, ds_ids, indegs)

        call build_flowtree_topology( &
            valids, ds_ids, indegs, nrows, ncols, &
            topo_order, topo_cnt, lvl_ends, nlvls, err_code)
        if (err_code /= 0) then
            deallocate (indegs, topo_order, lvl_ends)
            return
        end if

        call propagate_flowtree_metadata( &
            ds_ids, x, y, topo_order, lvl_ends, nlvls, nrows, ncols, &
            depths, sink_ids, sink_dists)

        deallocate (indegs, lvl_ends)
    end subroutine build_flowtree_metadata

    pure function find_flowtree_confluence(cid1, cid2, ds_ids, depths, jump_ids) &
        result(confluence_id)
        !! Returns the first common downstream cell using depth-block jumps.
        !!
        !! depth is measured from a cell downstream to its sink. jump_ids(v)
        !! identifies the anchor at the top of v's fixed-size depth block. The
        !! first loop skips whole blocks from whichever node has the deeper
        !! anchor. Once both nodes share an anchor, the second loop follows
        !! individual parent edges until they meet.
        !!
        !! For maximum tree depth D and block size B this changes a worst-case
        !! O(D) parent walk into approximately O(D/B + B), while needing only
        !! one jump integer per cell rather than the O(N log D) storage required
        !! by binary lifting.
        !!
        !! Precondition: cell 1 and cell 2 belong to the same sink tree. The caller
        !! establishes this without a per-query sink comparison: every pair of
        !! adjacent cells in different trees has both endpoints marked as basin
        !! boundary cells and is skipped before calling this function.
        implicit none(type, external)
        integer, intent(in) :: cid1, cid2
            !! Linear IDs of the two cells to process.
        integer, intent(in) :: ds_ids(:)
            !! Immediate downstream parent ID for every cell; zero at sinks.
        integer, intent(in) :: depths(:)
            !! Number of downstream edges between each cell and its sink.
        integer, intent(in) :: jump_ids(:)
            !! Depth-block anchor ID used to skip groups of parent edges.
        integer :: confluence_id
            !! Linear ID of the first common downstream cell; zero on invalid input.
        integer :: pid1, pid2
            !! Mutable downstream cursors used while aligning and joining paths.

        confluence_id = 0
        if (cid1 < 1 .or. cid1 > size(ds_ids)) return
        if (cid2 < 1 .or. cid2 > size(ds_ids)) return
        pid1 = cid1
        pid2 = cid2
        do while (jump_ids(pid1) /= jump_ids(pid2))
            if (depths(jump_ids(pid1)) > depths(jump_ids(pid2))) then
                pid1 = ds_ids(jump_ids(pid1))
            else
                pid2 = ds_ids(jump_ids(pid2))
            end if
        end do
        do while (pid1 /= pid2)
            ! Walk the branch farther from the sink
            if (depths(pid1) > depths(pid2)) then
                pid1 = ds_ids(pid1)
            else if (depths(pid2) > depths(pid1)) then
                pid2 = ds_ids(pid2)
            else
                pid1 = ds_ids(pid1)
                pid2 = ds_ids(pid2)
            end if
        end do
        confluence_id = pid1
    end function find_flowtree_confluence

    subroutine compute_max_branch_dist( &
        maxbdists, dirs, valids, x, y, nrows, ncols, &
        offsets, codes, noffsets, err_code)
        !! Computes, for every valid cell, the largest distance from
        !! that cell to its first downstream confluence with any of
        !! its eight neighbours. If a neighbour belongs to another
        !! sink tree, the two paths never converge and the cell's
        !! complete distance to its sink is considered.
        !!
        !! The implementation has four phases:
        !!
        !!  1. Build the downstream forest and cumulative sink
        !!     metadata.
        !!  2. Mark cells touching a different sink tree. Their
        !!     answer is known
        !!     immediately to be their complete sink distance.
        !!  3. Reuse the no-longer-needed sink-ID array for depth-
        !!     block jump
        !!     pointers used by lowest-common-ancestor searches.
        !!  4. Examine each undirected neighbour edge once and
        !!     atomically update the maximum for its two endpoints.
        !!
        !! The tree representation avoids tracing two complete flow
        !! paths for every neighbour pair. It also uses shared O(N)
        !! metadata rather than a full-grid visited/path workspace
        !! for every OpenMP thread.
        implicit none(type, external)
        ! Inputs
        integer, intent(in) :: nrows, ncols
            !! Size of the grid
        integer(c_int8_t), intent(in) :: dirs(nrows, ncols)
            !! Flow direction grid, using the provided codes
        real, intent(in) :: x(nrows, ncols), y(nrows, ncols)
            !! Grids of x and y coordinates for each cell, used to calculate distances between cells
        logical(kind=1), intent(in) :: valids(nrows, ncols)
            !! Validity mask (false for no-data)
        integer, intent(in) :: noffsets
            !! Number of flow directions
        integer, intent(in) :: offsets(noffsets, 2)
            !! List of offsets for each flow direction
        integer(c_int8_t), intent(in) :: codes(noffsets)
            !! List of flow direction codes corresponding to the offsets
        ! Outputs
        real, intent(out) :: maxbdists(nrows, ncols)
            !! Maximum downstream distance from each cell to its first
            !! confluence with any neighbouring cell.
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: The valid flow field contains a cycle
            !!   - 2: Internal workspace allocation failed
            !!   - 3: A traversal queue exceeded its allocated capacity
        ! Local variables
        integer, allocatable :: offset_lookup(:, :)
            !! Row/column displacement indexed directly by unsigned code.
        integer, allocatable :: ds_ids(:)
            !! Immediate downstream parent ID for every raster cell.
        integer, allocatable :: depths(:)
            !! Number of downstream edges from each cell to its sink.
        integer, allocatable :: sink_ids(:)
            !! Initially the sink root IDs; later reused as LCA jump pointers.
        integer, allocatable :: topo_order(:)
            !! Valid cell IDs ordered from upstream sources towards sinks.
        real, allocatable :: sink_dists(:)
            !! Cumulative metric distance from each cell to its sink.
        logical(kind=1), allocatable :: is_boundary(:)
            !! True when a cell touches a valid cell belonging to another sink.
        real :: dist1, dist2
            !! Branch distances from the two endpoints of the current grid edge.
        integer :: nneighbour
            !! Index of the neighbour orientation currently being evaluated.
        integer :: neighbour_offsets(4, 2)
            !! Half of the eight-neighbour stencil. These four orientations enumerate every undirected adjacency exactly once.
        integer :: boundary_offsets(8, 2)
            !! Complete eight-neighbour stencil used for boundary classification.
        ! A smaller block shortens the final parent walk but requires more block
        ! jumps. Sixteen was the fastest measured value on the representative
        ! elevation raster and adds no storage of its own.
        integer, parameter :: jump_block_size = 16
            !! Number of depth levels represented by one LCA jump block.
        integer :: ci, cj, ni, nj
            !! Row/column coordinates of the current and neighbouring cells.
        integer :: cid, nid
            !! Linear IDs of the current and neighbouring cells.
        integer :: conf_id
            !! Linear ID of the first common downstream cell for the current pair.
        integer :: alloc_stat
            !! Status returned by workspace allocation statements.
        integer :: topo_cnt
            !! Number of valid entries in topo_order.
        integer :: sorder
            !! Reverse topological cursor used while constructing jump pointers.
        logical :: on_border
            !! True when the current cell needs explicit neighbour bounds checks.

        ! Fortran reshape fills the first dimension first, producing the rows
        ! (1,1), (-1,1), (0,1), and (1,0). Together these cover every undirected
        ! eight-neighbour edge once. The opposite directions would duplicate
        ! both the confluence work and atomic output updates.
        parameter(neighbour_offsets= &
                  reshape([1, -1, &
                           0, 1, &
                           1, 1, &
                           1, 0 &
                           ], [4, 2]))
        ! Boundary classification assigns a property to the current cell only,
        ! so it requires the complete eight-neighbour stencil.
        parameter(boundary_offsets= &
                  reshape([-1, 0, 1, -1, 1, -1, 0, 1, &
                           -1, -1, -1, 0, 0, 1, 1, 1], [8, 2]))

        ! Create lookup tables for offsets
        err_code = 0
        if (nrows < 1 .or. ncols < 1) then
            err_code = 3
            maxbdists = 0.0
            return
        else if (ncols > huge(nrows)/nrows) then
            err_code = 3
            maxbdists = 0.0
            return
        end if
        allocate (offset_lookup(0:255, 2), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            return
        end if
        offset_lookup = fill_offset_lookup(offsets, codes)

        allocate (ds_ids(nrows*ncols), depths(nrows*ncols), &
                  sink_ids(nrows*ncols), sink_dists(nrows*ncols), &
                  is_boundary(nrows*ncols), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = 2
            deallocate (offset_lookup)
            return
        end if

        call build_flowtree_metadata( &
            dirs, valids, x, y, offset_lookup, nrows, ncols, &
            ds_ids, depths, sink_ids, sink_dists, &
            topo_order, topo_cnt, err_code)
        if (err_code /= 0) then
            deallocate (offset_lookup, ds_ids, depths, sink_ids, &
                        sink_dists, is_boundary)
            return
        end if

        ! Pre-fill cells on watershed boundaries: max branch distance is the distance to sink
        is_boundary = .false.
        maxbdists = 0.0
        !$omp PARALLEL DO DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, cid, nid, on_border) &
        !$omp COLLAPSE(2) SCHEDULE(STATIC)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                ! ci/cj are loop-controlled and already in bounds, making this
                ! unchecked column-major encoding safe.
                cid = ci + (cj - 1)*nrows
                ! Every stencil offset has magnitude <= 1. Interior cells can
                ! omit all per-neighbour bounds comparisons; only cells on the
                ! thin outer border require explicit coordinate checks.
                on_border = ci == 1 .or. ci == nrows .or. cj == 1 .or. cj == ncols
                do nneighbour = 1, size(boundary_offsets, 1)
                    ni = ci + boundary_offsets(nneighbour, 1)
                    nj = cj + boundary_offsets(nneighbour, 2)
                    if (on_border) then
                        if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    end if
                    if (.not. valids(ni, nj)) cycle
                    ! ni/nj are either interior-safe or checked above, so an
                    ! invalid coordinate cannot wrap into a legitimate ID.
                    nid = ni + (nj - 1)*nrows
                    if (sink_ids(cid) == sink_ids(nid)) cycle
                    is_boundary(cid) = .true.
                    maxbdists(ci, cj) = sink_dists(cid)
                    exit
                end do
            end do
        end do
        !$omp END PARALLEL DO

        ! Reuse the sink such that the watershed is broken down every jump_block_size steps
        do sorder = topo_cnt, 1, -1
            cid = topo_order(sorder)
            nid = ds_ids(cid)
            if (nid == 0 .or. mod(depths(cid), jump_block_size) == 0) then
                ! Sinks and exact block boundaries anchor their own blocks.
                sink_ids(cid) = cid
            else
                ! Other cells share the anchor of their downstream parent.
                sink_ids(cid) = sink_ids(nid)
            end if
        end do
        deallocate (topo_order)

        ! Static scheduling is intentional. Unchunked dynamic scheduling created
        ! one runtime work assignment per collapsed grid cell and dominated the
        ! representative workload.
        !$omp PARALLEL DO DEFAULT(SHARED) &
        !$omp PRIVATE(ci, cj, ni, nj, nneighbour, cid, nid, on_border) &
        !$omp PRIVATE(conf_id, dist1, dist2) &
        !$omp SCHEDULE(STATIC) COLLAPSE(2)
        do cj = 1, ncols
            do ci = 1, nrows
                if (.not. valids(ci, cj)) cycle
                cid = ci + (cj - 1)*nrows
                on_border = ci == 1 .or. ci == nrows .or. cj == 1 .or. cj == ncols
                do nneighbour = 1, size(neighbour_offsets, 1)
                    ni = ci + neighbour_offsets(nneighbour, 1)
                    nj = cj + neighbour_offsets(nneighbour, 2)
                    if (on_border) then
                        if (array2d_oob(ni, nj, nrows, ncols)) cycle
                    end if
                    if (.not. valids(ni, nj)) cycle
                    nid = ni + (nj - 1)*nrows
                    ! Different-tree pairs necessarily have both flags set and
                    ! already contributed full sink distances in phase 2.
                    if (is_boundary(cid) .and. is_boundary(nid)) cycle

                    conf_id = find_flowtree_confluence( &
                              cid, nid, ds_ids, depths, sink_ids)
                    if (conf_id == 0) then
                        ! Defensive fallback for an invalid/no-confluence query.
                        dist1 = sink_dists(cid)
                        dist2 = sink_dists(nid)
                    else
                        dist1 = sink_dists(cid) - sink_dists(conf_id)
                        dist2 = sink_dists(nid) - sink_dists(conf_id)
                    end if
                    if (.not. is_boundary(cid)) then
                        ! An endpoint participates in several edge updates that
                        ! may be owned by different threads. Atomic MAX prevents
                        ! lost updates without per-thread full-grid result arrays.
                        !$omp ATOMIC UPDATE
                        maxbdists(ci, cj) = max(maxbdists(ci, cj), dist1)
                        !$omp END ATOMIC
                    end if
                    if (.not. is_boundary(nid)) then
                        !$omp ATOMIC UPDATE
                        maxbdists(ni, nj) = max(maxbdists(ni, nj), dist2)
                        !$omp END ATOMIC
                    end if
                end do
            end do
        end do
        !$omp END PARALLEL DO
        deallocate (offset_lookup, ds_ids, depths, &
                    sink_ids, sink_dists, is_boundary)
    end subroutine compute_max_branch_dist

    pure subroutine compute_confluence_dist( &
        dists, &
        s1ij, s2ij, dirs, x, y, &
        offset_lookup, check_flag, err_code)
        !! Traces flow paths from two seed cells downstream to
        !! compute their confluence distance.
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: s1ij(2), s2ij(2)
            !! (i, j) indices of the two seed cells from which to trace flow paths
        integer(c_int8_t), intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical(kind=1), intent(in), optional :: check_flag
            !! Whether to check for confluence at each step
        ! Outputs
        real, intent(out) :: dists(2)
            !! Distances from each seed ceel to the confluence cell (or to max path length if no confluence found)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A traced flow path contains a cycle
            !!   - 2: Internal workspace allocation failed
            !!   - 3: A traced flow path exceeded its allocated capacity
        ! Local variables
        logical(kind=1) :: check_flag_
        integer :: maxpathlen
            !! Maximum path length to search before giving up and assuming no confluence
            !! It should be large enough to allow confluence but prevent infinite loops in case of errors.
        integer :: id1, id2
            !! IDs of the first and second paths to check for confluence
            !! When incrementing, each ID is of 'maxpathlen' apart such that 'path1id + ilen' is unique, which allows for more efficient confluence lookup.
        integer, allocatable :: path1(:, :), path2(:, :), visited(:, :)

        maxpathlen = 4*(size(dirs, 1) + size(dirs, 2))
        id1 = 1
        ! Storing path step offset in visited grid requires id1 and id2 to have disjoint active value ranges.
        ! id1 uses values in [id1, id1 + maxpathlen - 1].
        ! id2 uses values in [id2, id2 + maxpathlen - 1].
        id2 = 1 + maxpathlen
        err_code = 0
        allocate (path1(2, maxpathlen), path2(2, maxpathlen), &
                  visited(size(dirs, 1), size(dirs, 2)), stat=err_code)
        if (err_code /= 0) then
            err_code = 2
            dists = 0.0
            return
        end if
        visited = 0

        if (present(check_flag)) then
            check_flag_ = check_flag
        else
            check_flag_ = .true.
        end if

        call inner_compute_confluence_dist( &
            dists, &
            s1ij(1), s1ij(2), s2ij(1), s2ij(2), dirs, x, y, offset_lookup, &
            maxpathlen, path1, path2, visited, id1, id2, &
            check_flag=check_flag_, err_code=err_code)
        deallocate (path1)
        deallocate (path2)
        deallocate (visited)
    end subroutine compute_confluence_dist

    pure subroutine inner_compute_confluence_dist( &
        dists, s1i, s1j, s2i, s2j, dirs, x, y, &
        offset_lookup, maxpathlen, path1, path2, visited, &
        id1, id2, check_flag, err_code)
        !! Inner routine for computing the confluence distance
        !! between two seed cells.
        !!
        !! The 'visited' grid tracks cell visits. It stores the
        !! exact path step index: 'id + ipath - 1'. If
        !! 'visited(n1i, n1j)' is in the range
        !! [id2, id2 + npath2 - 1], it means path 2 has already
        !! visited this cell, and the index at which the confluence
        !! occurs in path 2 is then retrieved instantly via. This
        !! avoids an O(N) linear search over the path.
        implicit none(type, external)
        ! Inputs
        integer, intent(in) :: s1i, s1j, s2i, s2j
            !! Indices of the two seed cells from which to trace flow paths
        integer(c_int8_t), intent(in) :: dirs(:, :)
            !! Gird of flow direction codes and the corresponding offset lookup table
        real, contiguous, intent(in) :: x(:, :), y(:, :)
            !! Coordinates of cell centres for distance calculation
        integer, intent(in) :: offset_lookup(0:255, 2)
            !! Lookup table for offsets corresponding to each flow direction code, used to find downstream cell indices
        logical(kind=1), intent(in), optional :: check_flag
            !! Flag for whether to check for confluence at each step (can be turned off for performance if many confluences are expected)
        integer, intent(in) :: maxpathlen
            !! Maximum path length to search before giving up and assuming no confluence
            !! It should be large enough to allow confluence but prevent infinite loops in case of errors.
        integer, intent(in) :: id1, id2
            !! Unique ids to mark visited cells for each path in the visited grid
        integer, intent(inout) :: path1(2, maxpathlen), path2(2, maxpathlen)
            !! Workspace arrays for paths and visited grid
        integer, contiguous, intent(inout) :: visited(:, :)
            !! Grid to track visited paths by ids
        ! Outputs
        real, intent(out) :: dists(2)
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: Programme executed properly
            !!   - 1: A traced flow path contains a cycle
            !!   - 3: A traced flow path exceeded its allocated capacity
            !! Distances from each seed cell to the confluence cell (or to max path length if no confluence found)
        ! Local variables
        integer :: ipath1, ipath2, npath1, npath2
            !! Indices for iterating through paths and current path lengths
        integer :: iconf1, iconf2
            !! Indices of confluence in paths (or max path length if no confluence found)
        integer :: n1i, n1j, n2i, n2j
            !! Indices of next cell in path for each seed
        integer :: code1, code2
            !! Unsigned representations of flow direction codes for current cells in paths
        logical(kind=1) :: is_active1, is_active2, local_check_flag
            !! Flags for whether each path is still active (has not reached max length or invalid cell) and local copy of check_flag for performance

        !! Initialisation and checks
        if (present(check_flag)) then
            local_check_flag = check_flag
        else
            local_check_flag = .true.
        end if
        iconf1 = maxpathlen
        iconf2 = maxpathlen

        dists = 0.0
        err_code = 0
        is_active1 = .true.
        is_active2 = .true.

        ! Return zero if same cell
        if ((s1i == s2i) .and. (s1j == s2j)) then
            dists = 0.0
            return
        end if

        !! Main algorithm
        npath1 = 1
        path1(1, npath1) = s1i
        path1(2, npath1) = s1j
        visited(s1i, s1j) = id1
        npath2 = 1
        path2(1, npath2) = s2i
        path2(2, npath2) = s2j
        visited(s2i, s2j) = id2

        tracer_loop: do while (is_active1 .or. is_active2)
            path1_prc: block
                if (.not. is_active1) exit path1_prc
                ! Make sure code is valid
                code1 = iand(int(dirs(path1(1, npath1), path1(2, npath1))), 255)
                if (code1 < lbound(offset_lookup, 1) .or. code1 > ubound(offset_lookup, 1)) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                else if (offset_lookup(code1, 1) == 0 .and. offset_lookup(code1, 2) == 0) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                end if

                ! Compute next step
                n1i = path1(1, npath1) + offset_lookup(code1, 1)
                n1j = path1(2, npath1) + offset_lookup(code1, 2)
                if (array2d_oob(n1i, n1j, size(dirs, 1), size(dirs, 2))) then
                    iconf1 = npath1
                    is_active1 = .false.
                    exit path1_prc
                else if (npath1 >= maxpathlen) then
                    err_code = 3
                    iconf1 = npath1
                    return
                end if
                npath1 = npath1 + 1
                path1(1, npath1) = n1i
                path1(2, npath1) = n1j
                ! Check for self-intersection (value lies within Path 1's active range of IDs for the current run)
                if (visited(n1i, n1j) >= id1 .and. visited(n1i, n1j) < id1 + npath1 - 1) then
                    err_code = 1
                    iconf1 = npath1
                    return
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path1_prc
                ! If the cell wasn't visited by Path 2 in the current run (value not in Path 2's ID range),
                ! mark it with Path 1's base ID + step offset, and continue tracing.
                if (visited(n1i, n1j) < id2 .or. visited(n1i, n1j) >= id2 + npath2) then
                    visited(n1i, n1j) = id1 + npath1 - 1
                    exit path1_prc
                end if
                ! Confluence found: retrieve the exact matching index in Path 2 in O(1) time
                iconf1 = npath1
                iconf2 = visited(n1i, n1j) - id2 + 1
                exit tracer_loop
            end block path1_prc

            path2_prc: block
                if (.not. is_active2) exit path2_prc
                ! Make sure code is valid
                code2 = iand(int(dirs(path2(1, npath2), path2(2, npath2))), 255)
                if (code2 < lbound(offset_lookup, 1) .or. code2 > ubound(offset_lookup, 1)) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (offset_lookup(code2, 1) == 0 .and. offset_lookup(code2, 2) == 0) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                end if
                n2i = path2(1, npath2) + offset_lookup(code2, 1)
                n2j = path2(2, npath2) + offset_lookup(code2, 2)
                if (array2d_oob(n2i, n2j, size(dirs, 1), size(dirs, 2))) then
                    iconf2 = npath2
                    is_active2 = .false.
                    exit path2_prc
                else if (npath2 >= maxpathlen) then
                    err_code = 3
                    iconf2 = npath2
                    return
                end if
                npath2 = npath2 + 1
                path2(1, npath2) = n2i
                path2(2, npath2) = n2j
                ! Check for self-intersection (value lies within Path 2's active range of IDs for the current run)
                if (visited(n2i, n2j) >= id2 .and. visited(n2i, n2j) < id2 + npath2 - 1) then
                    err_code = 1
                    iconf2 = npath2
                    return
                end if
                ! Check if enters a visited cell
                if (.not. local_check_flag) exit path2_prc
                ! If the cell wasn't visited by Path 1 in the current run (value not in Path 1's ID range),
                ! mark it with Path 2's base ID + step offset, and continue tracing.
                if (visited(n2i, n2j) < id1 .or. visited(n2i, n2j) >= id1 + npath1) then
                    visited(n2i, n2j) = id2 + npath2 - 1
                    exit path2_prc
                end if
                ! Confluence found: retrieve the exact matching index in Path 1 in O(1) time
                iconf1 = visited(n2i, n2j) - id1 + 1
                iconf2 = npath2
                exit tracer_loop
            end block path2_prc
        end do tracer_loop

        ! Compute distances to confluence
        do ipath1 = 1, min(iconf1, npath1) - 1
            dists(1) = dists(1) + l2dist_xy( &
                       x(path1(1, ipath1 + 1), path1(2, ipath1 + 1)), &
                       y(path1(1, ipath1 + 1), path1(2, ipath1 + 1)), &
                       x(path1(1, ipath1), path1(2, ipath1)), &
                       y(path1(1, ipath1), path1(2, ipath1)))
        end do
        do ipath2 = 1, min(iconf2, npath2) - 1
            dists(2) = dists(2) + l2dist_xy( &
                       x(path2(1, ipath2 + 1), path2(2, ipath2 + 1)), &
                       y(path2(1, ipath2 + 1), path2(2, ipath2 + 1)), &
                       x(path2(1, ipath2), path2(2, ipath2)), &
                       y(path2(1, ipath2), path2(2, ipath2)))
        end do
    end subroutine inner_compute_confluence_dist
end module drainage_ridges
