!> Triangulates raster-grid vertices using the FORTRAN backend.
!!
!! This internal module implements incremental Bowyer-Watson
!! triangulation for integer 2D coordinates. Internal coordinates
!! use 64-bit integers so the super-triangle can enclose any int32
!! input. Triangle and vertex IDs are 1-based internally; the Python
!! wrapper returns 0-based IDs.
!!
!! Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
!! Last modified: 2026-08-13, En-Chi Lee (williameclee@gmail.com)

module meshing_triangulation
    use iso_c_binding, only: c_int32_t, c_int64_t
    use utils, only: ERR_NO_ERROR, ERR_ALLOCATION_FAILURE, &
                     ERR_OVERFLOW, ERR_COMPUTATION_FAILURE
    use intersections, only: incircle, orient_v2
    private :: make_initial_facets, insert_vertex, toggle_edge
    private :: find_triangle_side_neighbour
contains
    pure subroutine make_initial_facets(vtxs, facets, seeds, iinf, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! 2D index coordinates of the vertices.
        ! Outputs
        integer(c_int32_t), intent(out) :: facets(:, :)
        integer(c_int32_t), intent(out) :: seeds(3)
        integer(c_int32_t), intent(out) :: iinf
        integer, intent(out) :: err_code
        ! Local variables
        integer(c_int32_t) :: ivtx, jvtx
        integer(c_int32_t) :: orient
        integer(c_int32_t) :: facet(3)

        err_code = 0

        ! Find the first facet
        ivtx = 0
        do jvtx = 3, size(vtxs, 2)
            orient = orient_v2(vtxs(:, 1), vtxs(:, 2), vtxs(:, jvtx))
            if (orient /= 0) then
                ivtx = jvtx
                exit
            end if
        end do

        if (ivtx == 0) then
            ! All vertices are collinear
            err_code = ERR_COMPUTATION_FAILURE
            return
        elseif (size(facets, 2) < 4) then
            ! Not enough space to store all the triangles
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        seeds = [1, 2, ivtx]
        iinf = size(vtxs, 2) + 1

        if (orient > 0) then
            facet = [1, 2, ivtx]
        else
            facet = [1, ivtx, 2]
        end if
        facets(:, 1) = facet
        facets(:, 2) = [iinf, facet(2), facet(1)]
        facets(:, 3) = [iinf, facet(3), facet(2)]
        facets(:, 4) = [iinf, facet(1), facet(3)]
    end subroutine make_initial_facets

    pure logical function is_bad_facet(tri, ivtx, vtxs, iinf) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
        integer(c_int32_t), intent(in) :: tri(3)
        integer(c_int32_t), intent(in) :: ivtx
        integer(c_int32_t), intent(in) :: iinf
        ! Local variables
        integer :: inf_cnt
        integer :: jinf, jvtx, kvtx
        integer :: orient

        inf_cnt = count(tri == iinf)

        ! Base case: no infinite vertex
        if (inf_cnt == 0) then
            flag = incircle( &
                   vtxs(:, tri(1)), vtxs(:, tri(2)), vtxs(:, tri(3)), &
                   vtxs(:, ivtx)) > 0
            return
        end if

        ! Special case: has infinite vertex
        ! Find where the infinite vertex is, and rotate the
        ! triangle
        if (tri(1) == iinf) jinf = 1
        if (tri(2) == iinf) jinf = 2
        if (tri(3) == iinf) jinf = 3
        jvtx = tri(modulo(jinf, 3) + 1)
        kvtx = tri(modulo(jinf + 1, 3) + 1)

        orient = orient_v2(vtxs(:, jvtx), vtxs(:, kvtx), vtxs(:, ivtx))
        if (orient > 0) then
            flag = .true.
        elseif (orient == 0) then
            if (min(vtxs(1, jvtx), vtxs(1, kvtx)) <= vtxs(1, ivtx) .and. &
                vtxs(1, ivtx) <= max(vtxs(1, jvtx), vtxs(1, kvtx)) .and. &
                min(vtxs(2, jvtx), vtxs(2, kvtx)) <= vtxs(2, ivtx) .and. &
                vtxs(2, ivtx) <= max(vtxs(2, jvtx), vtxs(2, kvtx))) then
                flag = .true.
            else
                flag = .false.
            end if
        else
            flag = .false.
        end if
    end function is_bad_facet

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
                ! The edge could be stored in either orientation (but should be the opposite of the current one)
                if (min(edges(1, iedge), edges(2, iedge)) /= jvtx) cycle
                if (max(edges(1, iedge), edges(2, iedge)) /= kvtx) cycle
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
        ! Store in the CCW direction
        edges(:, nedges) = [vtx1, vtx2]
    end subroutine toggle_edge

    !> Inserts one vertex using the Bowyer-Watson cavity operation.
    !! Triangles whose circumcircles contain the vertex are replaced
    !! by counterclockwise triangles joining the vertex to the
    !! cavity boundary.
    pure subroutine insert_vertex( &
        ivtx, vtxs, triangles, ntris, iinf, &
        bad_tri_ids, edges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ivtx
            !! ID of the vertex in 'vtxs' to insert.
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
            !! This should include the vertices of the super-
            !! triangle.
        integer(c_int32_t), intent(in) :: iinf
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
            if (.not. is_bad_facet(triangles(:, itri), ivtx, vtxs, iinf)) cycle
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

            ! Should already be oriented CCW?
            if ((count([jvtx, kvtx, ivtx] == iinf) >= 1) .or. (orient > 0)) then
                triangles(:, itri) = [jvtx, kvtx, ivtx]
                ! else if (orient < 0) then
                !     triangles(:, itri) = [kvtx, jvtx, ivtx]
            else
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do
    end subroutine insert_vertex

    !> Triangulates unique integer vertices using incremental
    !! Bowyer-Watson.
    pure subroutine triangulate_points( &
        nvtxs, vtxs, facets, ntris, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the input
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D index coordinates of the vertices
        ! Outputs
        integer(c_int32_t), intent(out) :: facets(3, nvtxs*2 + 16)
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
        integer :: alloc_stat
        integer :: ivtx, itri
        integer(c_int32_t) :: seeds(3)
        integer(c_int32_t) :: iinf
        integer, allocatable :: bad_tri_ids(:)
            !! IDs of triangles in the current insertion cavity.
        integer, allocatable :: edges(:, :)
            !! Canonical cavity-boundary edge workspace.

        err_code = ERR_NO_ERROR

        allocate (bad_tri_ids(size(facets, 2)), &
                  edges(2, size(vtxs, 2)), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Make the first triangles
        call make_initial_facets(vtxs, facets, seeds, iinf, err_code)
        if (err_code /= ERR_NO_ERROR) return
        ntris = 4

        ! Insert vertices one at a time
        do ivtx = 1, nvtxs
            ! Skip already-processed seed vertices
            if (any(seeds == ivtx)) cycle
            call insert_vertex( &
                ivtx, vtxs, facets, ntris, iinf, &
                bad_tri_ids, edges, err_code)
            if (err_code /= ERR_NO_ERROR) return
        end do

        ! Remove unneeded triangles connected to the infinite vertex
        itri = 1
        do while (itri <= ntris)
            if (any(facets(:, itri) == iinf)) then
                facets(:, itri) = facets(:, ntris)
                ntris = ntris - 1
            else
                itri = itri + 1
            end if
        end do

        if (ntris <= 0) err_code = ERR_COMPUTATION_FAILURE
    end subroutine triangulate_points

    pure subroutine find_triangle_side_neighbour( &
        ivtx, jvtx, itri, iside, neighbours, edges, nedges, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: ivtx, jvtx, itri
            !! Vertex indices of the triangles
        integer, intent(in) :: iside
        integer(c_int32_t), intent(inout) :: neighbours(:, :)
        integer(c_int32_t), intent(inout) :: edges(:, :)
        integer, intent(inout) :: nedges
        integer, intent(inout) :: err_code
        ! Local variables
        integer :: iedge
        logical :: found_edge

        ! Find if the edge is already in the buffer
        found_edge = .false.
        if (nedges > 0) then
            do iedge = 1, nedges
                if (.not. ((edges(1, iedge) == ivtx) .and. &
                           (edges(2, iedge) == jvtx))) cycle
                found_edge = .true.
                exit
            end do
        end if

        if (found_edge) then
            neighbours(iside, itri) = edges(3, iedge)
            neighbours(edges(4, iedge), edges(3, iedge)) = itri
            ! Remove the edge since it is already found
            edges(:, iedge) = edges(:, nedges)
            nedges = nedges - 1
            return
        end if

        ! Insert the edge to the buffer
        if (nedges >= size(edges, 2)) then
            err_code = ERR_OVERFLOW
            return
        end if
        nedges = nedges + 1
        edges(1, nedges) = ivtx
        edges(2, nedges) = jvtx
        edges(3, nedges) = itri
        edges(4, nedges) = iside
    end subroutine find_triangle_side_neighbour

    pure subroutine find_triangle_neighbours( &
        ntris, triangles, neighbours, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ntris
            !! Number of triangles in the 'triangles' array.
        integer(c_int32_t), intent(in) :: triangles(3, ntris)
            !! Vertex indices of the triangles
        ! Outputs
        integer(c_int32_t), intent(out) :: neighbours(3, ntris)
        integer, intent(out) :: err_code
        ! Local variables
        integer :: itri
        integer :: ivtx, jvtx
        integer(c_int32_t), parameter :: no_neighbour = -1
        integer :: nedges
        integer(c_int32_t), allocatable :: edges(:, :)
        integer :: alloc_stat
        integer :: iside

        err_code = ERR_NO_ERROR
        allocate (edges(4, ntris*3), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        neighbours = no_neighbour
        nedges = 0
        do itri = 1, ntris
            do iside = 1, 3
                ! Skip if complement already found
                if (neighbours(iside, itri) /= no_neighbour) cycle

                ivtx = triangles(modulo(iside, 3) + 1, itri)
                jvtx = triangles(modulo(iside + 1, 3) + 1, itri)
                call find_triangle_side_neighbour( &
                    min(ivtx, jvtx), max(ivtx, jvtx), itri, iside, &
                    neighbours, edges, nedges, err_code)
                if (err_code /= ERR_NO_ERROR) return
            end do
        end do
    end subroutine find_triangle_neighbours
end module meshing_triangulation
