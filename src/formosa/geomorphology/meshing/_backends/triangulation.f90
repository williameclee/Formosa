!> Triangulates raster-grid vertices using the FORTRAN backend.
!!
!! This internal module implements incremental Bowyer-Watson
!! triangulation for integer 2D coordinates. Internal coordinates
!! use 64-bit integers so the super-triangle can enclose any int32
!! input. Triangle and vertex IDs are 1-based internally; the Python
!! wrapper returns 0-based IDs.
!!
!! Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)

module meshing_triangulation
    use iso_c_binding, only: c_int32_t, c_int64_t
    use utils, only: ERR_NO_ERROR, ERR_ALLOCATION_FAILURE, &
                     ERR_OVERFLOW, ERR_COMPUTATION_FAILURE
    use intersections, only: incircle, orient_v2
    private :: add_supertriangle, insert_vertex, toggle_edge
contains
    !> Copies the input vertices, appends a counterclockwise
    !! triangle that encloses their bounding box, and initialises
    !! the triangulation with that triangle.
    pure subroutine add_supertriangle( &
        vtxs, all_vtxs, triangles, ntris, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! 2D index coordinates of the vertices.
        ! Outputs
        integer(c_int64_t), intent(inout) :: all_vtxs(:, :)
            !! Input coordinates followed by the super-triangle.
        integer(c_int32_t), intent(inout) :: triangles(:, :)
            !! Vertex indices of the triangles.
        integer, intent(inout) :: ntris
            !! Number of triangles in the 'triangles' array.
        integer, intent(inout) :: err_code
            !! Shared backend status code
            !! - Set to OVERFLOW if no face slot is available for
            !! the initial triangle.
        ! Local variables
        integer :: nvtxs
            !! Number of vertices in the input
        integer(c_int64_t) :: minx, maxx, miny, maxy
        integer(c_int64_t) :: midx, xspan, yspan
            !! Bounding-box limits, integer centre, and maximum
            !! extents.

        nvtxs = size(vtxs, 2)
        ! Copy the points
        all_vtxs(:, 1:nvtxs) = int(vtxs, kind=c_int64_t)

        ! Add the supertriangle's vertices
        minx = minval(all_vtxs(1, 1:nvtxs))
        maxx = maxval(all_vtxs(1, 1:nvtxs))
        miny = minval(all_vtxs(2, 1:nvtxs))
        maxy = maxval(all_vtxs(2, 1:nvtxs))

        xspan = max(maxx - minx, 1_c_int64_t)
        yspan = max(maxy - miny, 1_c_int64_t)

        midx = minx + (maxx - minx)/2

        all_vtxs(:, nvtxs + 1) = &
            [midx - 3*xspan, miny - yspan]
        all_vtxs(:, nvtxs + 2) = &
            [midx + 3*xspan, miny - yspan]
        all_vtxs(:, nvtxs + 3) = &
            [midx, maxy + 2*yspan]

        ! Add the supertriangle's faces
        if (ntris >= size(triangles, 2)) then
            err_code = ERR_OVERFLOW
            return
        end if
        ntris = ntris + 1
        triangles(:, ntris) = [nvtxs + 1, nvtxs + 2, nvtxs + 3]
    end subroutine add_supertriangle

    !> Toggles a canonical edge in the cavity-edge buffer.
    !!
    !! The first occurrence adds the edge and the second removes it,
    !! leaving only boundary edges after every bad triangle has been
    !! processed.
    pure subroutine toggle_edge(vtx1, vtx2, edges, nedges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: vtx1, vtx2
            !! 1-based endpoint vertex IDs.
        integer, intent(inout) :: edges(:, :)
            !! Canonical endpoint pairs.
        integer, intent(inout) :: nedges
            !! Number of active columns in the edge buffer.
        integer, intent(inout) :: err_code
            !! Shared backend status code
            !! - Set to OVERFLOW when full.
        ! Local variables
        integer :: jvtx, kvtx
        integer :: iedge

        jvtx = min(vtx1, vtx2)
        kvtx = max(vtx1, vtx2)

        ! Find if the edge is already in the buffer
        if (nedges > 0) then
            do iedge = 1, nedges
                if (.not. ((edges(1, iedge) == jvtx) .and. &
                           (edges(2, iedge) == kvtx))) cycle
                ! A repeated edge is interior, remove from the buffer
                edges(:, iedge) = edges(:, nedges)
                nedges = nedges - 1
                return
            end do
        end if
        ! If not found, add it to the buffer
        if (nedges >= size(edges, 2)) then
            err_code = ERR_OVERFLOW
            return
        end if
        nedges = nedges + 1
        edges(1, nedges) = jvtx
        edges(2, nedges) = kvtx
    end subroutine toggle_edge

    !> Inserts one vertex using the Bowyer-Watson cavity operation.
    !! Triangles whose circumcircles contain the vertex are replaced
    !! by counterclockwise triangles joining the vertex to the
    !! cavity boundary.
    pure subroutine insert_vertex( &
        ivtx, vtxs, triangles, ntris, &
        bad_tri_ids, edges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ivtx
            !! ID of the vertex in 'vtxs' to insert.
        integer(c_int64_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
            !! This should include the vertices of the super-
            !! triangle.
        integer(c_int32_t), intent(inout) :: triangles(:, :)
            !! Vertex indices of the triangles
        integer, intent(inout) :: ntris
            !! Number of triangles actually in the 'triangles'
            !! array.
        integer, intent(inout) :: bad_tri_ids(size(triangles, 2))
            !! Workspace containing IDs of triangles in the cavity.
        integer, intent(inout) :: edges(2, size(vtxs, 2))
            !! Workspace containing canonical cavity-boundary edges.
        integer, intent(inout) :: err_code
            !! Shared backend status code.
        ! Local variables
        integer :: itri, ibadtri
        integer :: nbadtris
        integer :: nedges, iedge
        integer :: jvtx, kvtx
            !! Endpoint IDs of the current cavity edge.
        integer(c_int64_t) :: orient
            !! Signed orientation determinant for a candidate
            !! triangle.

        err_code = ERR_NO_ERROR

        ! Find triangles whose circumcircle contains the new vertex
        nbadtris = 0
        do itri = 1, ntris
            if (incircle(vtxs(:, triangles(1, itri)), &
                         vtxs(:, triangles(2, itri)), &
                         vtxs(:, triangles(3, itri)), &
                         vtxs(:, ivtx)) <= 0) cycle
            nbadtris = nbadtris + 1
            bad_tri_ids(nbadtris) = itri
        end do
        if (nbadtris <= 0) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        ! Add bad edges to the buffer
        nedges = 0
        do ibadtri = 1, nbadtris
            itri = bad_tri_ids(ibadtri)
            call toggle_edge( &
                triangles(1, itri), triangles(2, itri), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
            call toggle_edge( &
                triangles(2, itri), triangles(3, itri), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
            call toggle_edge( &
                triangles(3, itri), triangles(1, itri), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
        end do

        ! Make triangles from the boundary edges
        do iedge = 1, nedges
            jvtx = edges(1, iedge)
            kvtx = edges(2, iedge)

            ! Find where to insert the new triangle
            if (iedge <= nbadtris) then
                itri = bad_tri_ids(iedge)
            else
                if (ntris >= size(triangles, dim=2)) then
                    err_code = ERR_OVERFLOW
                    return
                end if
                ntris = ntris + 1
                itri = ntris
            end if
            ! Insert CCW triangle
            orient = orient_v2( &
                     vtxs(:, jvtx), vtxs(:, kvtx), vtxs(:, ivtx))

            if (orient > 0) then
                triangles(:, itri) = [jvtx, kvtx, ivtx]
            else if (orient < 0) then
                triangles(:, itri) = [kvtx, jvtx, ivtx]
            else
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do
    end subroutine insert_vertex

    !> Triangulates unique integer vertices using incremental
    !! Bowyer-Watson.
    pure subroutine triangulate_points( &
        nvtxs, vtxs, triangles, ntris, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the input
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D index coordinates of the vertices
        ! Outputs
        integer(c_int32_t), intent(out) :: triangles(3, nvtxs*2 + 16)
            !! Vertex indices of the triangles
        integer, intent(out) :: ntris
            !! Number of triangles actually in the 'triangles' array
        integer, intent(out) :: err_code
            !! Code indicating the status of the result
            !!   - 0: completed successfully
            !!   - 1: invalid input
            !!   - 2: workspace allocation failed
            !!   - 3: triangle or edge capacity exceeded
            !!   - 4: invalid or degenerate triangulation data
        ! Local variables
        integer :: ivtx, itri
        integer :: alloc_stat
        integer(c_int64_t), allocatable :: all_vtxs(:, :)
            !! Input and super-triangle coordinates.
        integer, allocatable :: bad_tri_ids(:)
            !! IDs of triangles in the current insertion cavity.
        integer, allocatable :: edges(:, :)
            !! Canonical cavity-boundary edge workspace.

        err_code = ERR_NO_ERROR

        allocate (all_vtxs(2, nvtxs + 3), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        allocate (bad_tri_ids(size(triangles, 2)), &
                  edges(2, size(all_vtxs, 2)), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Make the first super triangle
        ntris = 0
        call add_supertriangle( &
            vtxs, all_vtxs, triangles, ntris, err_code)
        if (err_code /= ERR_NO_ERROR) return

        ! Insert vertices one at a time
        do ivtx = 1, nvtxs
            call insert_vertex( &
                ivtx, all_vtxs, triangles, ntris, &
                bad_tri_ids, edges, err_code)
            if (err_code /= ERR_NO_ERROR) return
        end do

        ! Remove unneeded triangles connected to the super triangle
        itri = 1
        do while (itri <= ntris)
            if (any(triangles(:, itri) > nvtxs)) then
                triangles(:, itri) = triangles(:, ntris)
                ntris = ntris - 1
            else
                itri = itri + 1
            end if
        end do

        if (ntris <= 0) err_code = ERR_COMPUTATION_FAILURE
    end subroutine triangulate_points
end module meshing_triangulation
