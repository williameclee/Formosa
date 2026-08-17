!> Triangulates raster-grid vertices using the Fortran backend.
!!
!! This internal module implements incremental Bowyer-Watson
!! triangulation for integer 2D coordinates. Internal coordinates
!! use 64-bit integers so the super-triangle can enclose any int32
!! input. Triangle and vertex IDs are 1-based internally; the Python
!! wrapper returns 0-based IDs.
!!
!! Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
!! Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)

module meshing_triangulation
    use iso_c_binding, only: c_int32_t, c_int64_t
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW, &
                     ERR_COMPUTATION_FAILURE
    use utils, only: mod1, modshift
    use intersections, only: incircle, orient_v2, xcross, xcross_orient
    private :: make_initial_facets, insert_vertex, toggle_edge
    private :: find_triangle_side_neighbour
    private :: update_flipped_neighbours
    private :: edge_locked, edge_match, update_edge_record
    private :: remove_crossing, restore_deluanay_triangulation
    ! Moule variables
    integer(c_int32_t), parameter :: no_neighbour = -1
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

        err_code = ERR_NO_ERROR

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

    !> Tests whether a triangle belongs to an insertion cavity.
    !!
    !! Finite triangles use the in-circle predicate. Facets
    !! containing the symbolic infinite vertex use hull visibility
    !! instead.
    pure logical function is_bad_facet(tri, ivtx, vtxs, iinf) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
        integer(c_int32_t), intent(in) :: tri(3)
            !! Triangle vertex IDs in counterclockwise order.
        integer(c_int32_t), intent(in) :: ivtx
            !! ID of the candidate vertex being inserted.
        integer(c_int32_t), intent(in) :: iinf
            !! Symbolic infinite-vertex ID.
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
                ! The edge could be stored in either orientation
                !! (but should be the opposite of the current one)
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
    !!
    !! The returned triangles use counterclockwise, one-based vertex
    !! IDs. Their column order is not canonicalised by this routine.
    pure subroutine triangulate_points( &
        nvtxs, vtxs, facets, ntris, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the input.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! Unique 2-D integer coordinates of the vertices.
        ! Outputs
        integer(c_int32_t), intent(out) :: facets(3, nvtxs*2 + 16)
            !! Counterclockwise, one-based triangle vertex IDs.
        integer, intent(out) :: ntris
            !! Number of active columns in 'facets'.
        integer, intent(out) :: err_code
            !! Shared backend status code:
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

    !> Finds the adjacent triangle across each triangle side.
    !!
    !! Side 'i' is opposite vertex 'i'. Boundary sides receive the
    !! sentinel value 'no_neighbour'.
    pure subroutine find_triangle_neighbours( &
        ntris, triangles, neighbours, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ntris
            !! Number of triangles in the mesh.
        integer(c_int32_t), intent(in) :: triangles(3, ntris)
            !! 1-based triangle vertex IDs.
        ! Outputs
        integer(c_int32_t), intent(out) :: neighbours(3, ntris)
            !! 1-based adjacent triangle IDs across corresponding
            !! sides, or 'no_neighbour' at the mesh boundary.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !!   - 0: completed successfully
            !!   - 2: edge-workspace allocation failed
            !!   - 3: edge-workspace capacity exceeded
        ! Local variables
        integer :: itri
        integer :: ivtx, jvtx
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

    pure subroutine update_flipped_neighbours( &
        nabrs, itri, iside, jtri, jside)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(inout) :: nabrs(:, :)
        integer, intent(in) :: itri, iside, jtri, jside
        ! Local variables
        integer(c_int32_t) :: inabrs(3), jnabrs(3)
        integer :: innabr, jnnabr
        integer :: inside, inside_i, inside_j

        ! Preserve both rows and locate reciprocal outside-neighbour
        ! entries before changing the neighbour table.
        inabrs = nabrs(:, itri)
        jnabrs = nabrs(:, jtri)
        innabr = jnabrs(modshift(jside, 1, 3))
        jnnabr = inabrs(modshift(iside, 1, 3))
        inside_i = 0
        inside_j = 0
        if (innabr /= no_neighbour) then
            do inside = 1, 3
                if (nabrs(inside, innabr) == jtri) then
                    inside_i = inside
                    exit
                end if
            end do
        end if
        if (jnnabr /= no_neighbour) then
            do inside = 1, 3
                if (nabrs(inside, jnnabr) == itri) then
                    inside_j = inside
                    exit
                end if
            end do
        end if

        ! Change the two incident triangles.
        nabrs(:, itri) = &
            [jnabrs(modshift(jside, 1, 3)), jtri, &
             inabrs(modshift(iside, 2, 3))]
        nabrs(:, jtri) = &
            [jnabrs(modshift(jside, 2, 3)), &
             inabrs(modshift(iside, 1, 3)), itri]

        ! Update the reciprocal entries in the outside neighbours.
        if (inside_i > 0) nabrs(inside_i, innabr) = itri
        if (inside_j > 0) nabrs(inside_j, jnnabr) = jtri
    end subroutine update_flipped_neighbours

    !> Tests if a quadrilateral span by
    !! ***a***-***c***-***b***-***d*** (i.e. ***a*** & ***b*** are
    !! opposites, and so are ***c*** & ***d***) is convex.
    pure logical function is_convex(a, b, c, d) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: a(2), b(2), c(2), d(2)
            !! Quadrilateral coordinates, with 'a' opposite 'b' and
            !! 'c' opposite 'd'.
        ! Local variables
        integer(c_int64_t) :: orient_u, orient_v

        orient_u = orient_v2(a, b, c)
        orient_v = orient_v2(a, b, d)

        if ((orient_u == 0) .or. (orient_v == 0)) then
            ! Degenerate triangle (collinear)
            flag = .false.
            return
        elseif ((orient_u > 0) .eqv. (orient_v > 0)) then
            ! Not convex, or they should have opposite signs
            flag = .false.
            return
        end if

        flag = .true.
    end function is_convex

    !> Finds the triangle and local side sharing a mesh edge.
    !!
    !! The input side must be interior and have a reciprocal entry
    !! in the neighbour table.
    pure subroutine find_edge_sharing_triangle( &
        nabrs, itri, iside, jtri, jside, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: nabrs(:, :)
            !! 1-based triangle neighbours, with 'no_neighbour' at
            !! the mesh boundary.
        integer, intent(in) :: itri, iside
            !! Triangle and local side identifying the input edge.
        integer, intent(out) :: jtri, jside
            !! Adjacent triangle and its reciprocal local side.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 4: the edge is a boundary edge or has no
            !!     reciprocal neighbour entry

        err_code = ERR_NO_ERROR
        jtri = nabrs(iside, itri)
        if (jtri < lbound(nabrs, 2) .or. jtri > ubound(nabrs, 2)) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if
        do jside = 1, 3
            if (nabrs(jside, jtri) == itri) then
                exit
            elseif (jside == 3) then
                ! No matching side found, something must be wrong
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do
    end subroutine find_edge_sharing_triangle

    !> Flips an interior triangle edge in a convex quadrilateral.
    !!
    !! The routine updates the two incident triangles, their
    !! neighbour records, and reciprocal records in adjacent
    !! triangles. The new edge is side 2 of triangle 'itri' and side
    !! 3 of the other triangle.
    pure subroutine flip_triangle_edge( &
        vtxs, triangles, nabrs, nvtxs, ntris, itri, iside, changes, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs, ntris
            !! Numbers of vertices and triangles in the mesh.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2-D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: triangles(3, ntris)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(inout) :: nabrs(3, ntris)
            !! 1-based triangle neighbours, updated in place.
        integer, intent(in) :: itri, iside
            !! Triangle and local side identifying the edge to flip.
        ! Outputs
        integer(c_int32_t), intent(out) :: changes(4, 4)
            !! Descriptors '[itri, iside, vtx1, vtx2]' for the four
            !! non-flipped sides whose ownership may have changed.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 1: 'itri' or 'iside' is out of bounds
            !! - 4: the edge is on the boundary, its neighbour
            !!     record is invalid, or the quadrilateral is not
            !!     convex
        ! Local variables
        integer :: jtri, jside
        integer(c_int32_t) :: p, q, u, v

        err_code = ERR_NO_ERROR

        if ((itri < 1) .or. (itri > ntris)) then
            err_code = ERR_INVALID_INPUT
            return
        elseif ((iside < 1) .or. (iside > 3)) then
            err_code = ERR_INVALID_INPUT
            return
        end if

        ! Find the triangle/side sharing the edge
        call find_edge_sharing_triangle(nabrs, itri, iside, jtri, jside, err_code)
        if (err_code /= ERR_NO_ERROR) return

        ! Find the vertices
        p = triangles(iside, itri)
        q = triangles(jside, jtri)
        u = triangles(modshift(iside, 1, 3), itri)
        v = triangles(modshift(iside, 2, 3), itri)

        ! Check the edge is actually flippable
        if (.not. is_convex(vtxs(:, p), vtxs(:, q), vtxs(:, u), vtxs(:, v))) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        ! Flip the triangles
        triangles(:, itri) = [p, u, q]
        triangles(:, jtri) = [p, q, v]
        ! Update the neighbours
        call update_flipped_neighbours( &
            nabrs, itri, iside, jtri, jside)

        ! Record the other changed edges
        changes(:, 1) = [itri, 1, min(u, q), max(u, q)]
        changes(:, 2) = [itri, 3, min(p, u), max(p, u)]
        changes(:, 3) = [jtri, 2, min(v, p), max(v, p)]
        changes(:, 4) = [jtri, 1, min(q, v), max(q, v)]
    end subroutine flip_triangle_edge

    pure subroutine update_edge_record(edges, nedges, changed_edges)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(inout) :: edges(:, :)
        integer, intent(in) :: nedges
        integer(c_int32_t), intent(in) :: changed_edges(:, :)
        ! Local variables
        integer :: iedge, icedge

        do iedge = 1, nedges
            do icedge = 1, size(changed_edges, dim=2)
                if (.not. (edges(3, iedge) == changed_edges(3, icedge) .and. &
                           edges(4, iedge) == changed_edges(4, icedge))) cycle
                edges(1:2, iedge) = changed_edges(1:2, icedge)
            end do
        end do
    end subroutine update_edge_record

    !> Finds unique interior edges properly crossing a constraint
    !! edge.
    !!
    !! Determines proper line-segment intersections between
    !! unique interior triangulation edges and a target constraint
    !! segment using 64-bit 2D cross-product orientation
    !! predicates.
    pure subroutine find_crossing_edges( &
        vtxs, triangles, nabrs, nvtxs, ntris, &
        edge, xngs, nxngs, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the triangulation.
        integer, intent(in) :: ntris
            !! Number of triangles in the triangulation.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D index coordinates of the vertices.
        integer(c_int32_t), intent(in) :: triangles(3, ntris)
            !! Vertex indices of the triangles.
        integer(c_int32_t), intent(in) :: nabrs(3, ntris)
            !! Triangle neighbour indices across sides.
        integer, intent(in) :: edge(2)
            !! 1-based endpoint vertex IDs of the constraint edge.
        ! Outputs
        integer(c_int32_t), intent(out) :: xngs(4, ntris)
            !! Descriptor columns [itri, iside, vtx1, vtx2] for
            !! crossing interior mesh edges.
        integer(c_int32_t), intent(out) :: nxngs
            !! Total number of crossing mesh edges found.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !!   - 0: completed successfully
            !!   - 2: workspace allocation failed
        ! Local variables
        integer :: itris(3*ntris), isides(3*ntris)
            !! Triangle and side IDs owning each unique interior
            !! edge.
        integer :: itri, iside, iedge, nedges, ixng
            !! Loop indices and counters for candidate and
            !! crossing edges.
        integer(c_int32_t), allocatable :: ia(:), ib(:)
            !! Endpoint vertex IDs of unique interior mesh edges.
        integer(c_int64_t) :: u(2), v(2), uv(2)
            !! Coordinates of constraint endpoints and target
            !! constraint vector.
        integer(c_int64_t), allocatable :: a(:, :), b(:, :)
            !! Endpoint coordinates of unique interior mesh edges.
        integer(c_int64_t), allocatable :: &
            ab(:, :), au(:, :), av(:, :), ua(:, :), ub(:, :)
            !! Difference vectors for 2D orientation calculations.
        integer(c_int64_t), allocatable :: &
            orient_uva(:), orient_uvb(:), &
            orient_abu(:), orient_abv(:)
            !! 2D cross-product orientation determinants.
        logical(kind=1), allocatable :: is_xng(:)
            !! Boolean mask identifying proper crossing edges.
        integer :: alloc_stat
            !! Dynamic allocation status code.

        err_code = ERR_NO_ERROR

        ! Extract unique interior edges
        nedges = 0
        do iedge = 1, ntris*3
            itri = (iedge - 1)/3 + 1
            iside = mod1(iedge, 3)
            if (nabrs(iside, itri) == no_neighbour) cycle
            if (itri >= nabrs(iside, itri)) cycle
            nedges = nedges + 1
            itris(nedges) = itri
            isides(nedges) = iside
        end do

        ! Fetch vertex coordinates and their distance vectors
        u = vtxs(:, edge(1))
        v = vtxs(:, edge(2))
        uv = v - u
        allocate (ia(nedges), ib(nedges), &
                  a(2, nedges), b(2, nedges), ab(2, nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        do iedge = 1, nedges
            itri = itris(iedge)
            iside = isides(iedge)
            ia(iedge) = triangles(modshift(iside, 1, 3), itri)
            ib(iedge) = triangles(modshift(iside, 2, 3), itri)
            a(:, iedge) = vtxs(:, ia(iedge))
            b(:, iedge) = vtxs(:, ib(iedge))
        end do

        ab = b - a

        allocate (ua(2, nedges), ub(2, nedges), &
                  orient_uva(nedges), orient_uvb(nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Compute orientation of edge endpoints relative to
        ! constraint vector
        ua = a - spread(u, dim=2, ncopies=nedges)
        ub = b - spread(u, dim=2, ncopies=nedges)
        orient_uva = uv(1)*ua(2, :) - uv(2)*ua(1, :)
        orient_uvb = uv(1)*ub(2, :) - uv(2)*ub(1, :)

        allocate (orient_abu(nedges), orient_abv(nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Compute orientation of constraint endpoints relative to
        ! mesh edge vectors
        call move_alloc(from=ua, to=au)
        call move_alloc(from=ub, to=av)
        ab = b - a
        au = spread(u, dim=2, ncopies=nedges) - a
        av = spread(v, dim=2, ncopies=nedges) - a
        orient_abu = ab(1, :)*au(2, :) - ab(2, :)*au(1, :)
        orient_abv = ab(1, :)*av(2, :) - ab(2, :)*av(1, :)

        ! Classify proper line-segment crossings (Xs) with strict
        ! opposite orientations
        allocate (is_xng(nedges), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        is_xng = xcross_orient(orient_uva, orient_uvb, orient_abu, orient_abv)
        nxngs = count(is_xng)

        ! Pack crossing edge descriptors into output matrix
        ixng = 0
        do iedge = 1, nedges
            if (.not. is_xng(iedge)) cycle
            ixng = ixng + 1
            xngs(:, ixng) = &
                [itris(iedge), isides(iedge), &
                 min(ia(iedge), ib(iedge)), max(ia(iedge), ib(iedge))]
        end do
    end subroutine find_crossing_edges

    !> Checks if an edge is locked (i.e. a constraint) and cannot be
    !! flipped.
    pure logical function edge_locked(edge, locked_edges, nedges)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: edge(2)
        integer(c_int32_t), intent(in) :: locked_edges(2, nedges)
        integer, intent(in) :: nedges
        integer(c_int32_t) :: lo, hi
        integer :: iedge

        lo = minval(edge)
        hi = maxval(edge)
        edge_locked = .false.

        do iedge = 1, nedges
            if (minval(locked_edges(:, iedge)) == lo .and. &
                maxval(locked_edges(:, iedge)) == hi) then
                edge_locked = .true.
                return
            end if
        end do
    end function edge_locked

    !> Checks if the vertices of a specific edge is the same as
    !! claimed.
    pure logical function edge_match(triangles, edge)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: triangles(:, :)
        integer, intent(in) :: edge(4)
        ! Local variables
        integer :: ar, br
            !! Actual vertex indices for the iside-th edge of the
            !! itri-th triangle.

        ar = triangles(modshift(edge(2), 1, 3), edge(1))
        br = triangles(modshift(edge(2), 2, 3), edge(1))
        edge_match = (edge(3) == ar .and. edge(4) == br) .or. &
                     (edge(3) == br .and. edge(4) == ar)
    end function edge_match

    pure subroutine remove_crossing( &
        vtxs, faces, nabrs, nvtxs, ntris, edge, &
        xngs, nxngs, ixng, new_edges, nedges, nfailed, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
        integer, intent(in) :: ntris
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
        integer(c_int32_t), intent(inout) :: faces(3, ntris)
        integer(c_int32_t), intent(inout) :: nabrs(3, ntris)
        integer(c_int32_t), intent(in) :: edge(2)
        integer(c_int32_t), intent(inout) :: xngs(4, ntris)
        integer, intent(inout) :: nxngs, ixng
        integer(c_int32_t), intent(inout) :: new_edges(4, ntris)
        integer, intent(inout) :: nedges
        integer, intent(inout) :: nfailed
        integer, intent(out) :: err_code
        ! Local variables
        integer :: iface, iside, jface, jside
        integer(c_int32_t) :: vk(2), vl(2), vm(2), vn(2)
        integer :: k, l, m, n
        integer(c_int32_t) :: new_edge(4)
        integer(c_int32_t) :: changed_edges(4, 4)

        err_code = ERR_NO_ERROR

        if (.not. edge_match(faces, xngs(:, ixng))) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if
        ! Get the coordinates
        iface = xngs(1, ixng)
        iside = xngs(2, ixng)
        k = xngs(3, ixng)
        l = xngs(4, ixng)
        vk = vtxs(:, k)
        vl = vtxs(:, l)
        ! Note: 'nabrs' ordered such that the i-th edge is
        ! composed of the j-th and k-th vertices of the facet
        m = faces(iside, iface)
        vm = vtxs(:, m)
        call find_edge_sharing_triangle(nabrs, iface, iside, jface, jside, err_code)
        if (err_code /= ERR_NO_ERROR) return
        n = faces(jside, jface)
        vn = vtxs(:, n)

        ! Swap edge if quadrilateral is convex
        if (.not. is_convex(vm, vn, vk, vl)) then
            nfailed = nfailed + 1
            ! If looped through all edges and none is flippable
            if (nfailed >= nxngs) then
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
            return
        end if
        call flip_triangle_edge( &
            vtxs, faces, nabrs, nvtxs, ntris, iface, iside, &
            changed_edges, err_code)
        if (err_code /= ERR_NO_ERROR) return
        ! Update potentially changed edges
        call update_edge_record(xngs, nxngs, changed_edges)
        call update_edge_record(new_edges, nedges, changed_edges)

        ! Replace the original crossing
        ! The new side is always the 2nd side of the triangle
        new_edge = [iface, 2, min(m, n), max(m, n)]
        nfailed = 0

        ! Check if the new edge is the constraint, or if it still crosses the constraint
        if (new_edge(3) == minval(edge) .and. new_edge(4) == maxval(edge)) then
            nedges = nedges + 1
            new_edges(:, nedges) = new_edge
            return
        elseif (xcross(vtxs(:, edge(1)), vtxs(:, edge(2)), vm, vn)) then
            xngs(:, ixng) = new_edge
        else
            ! Remove the crossing
            xngs(:, ixng) = xngs(:, nxngs)
            nxngs = nxngs - 1
            if (nxngs > 0) ixng = modshift(ixng, -1, nxngs)
            ! Record the new edge
            nedges = nedges + 1
            new_edges(:, nedges) = new_edge
        end if
    end subroutine remove_crossing

    pure subroutine restore_deluanay_triangulation( &
        vtxs, faces, nabrs, nvtxs, ntris, edge, edges, nedges, &
        err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
        integer, intent(in) :: ntris
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
        integer(c_int32_t), intent(inout) :: faces(3, ntris)
        integer(c_int32_t), intent(inout) :: nabrs(3, ntris)
        integer(c_int32_t), intent(in) :: edge(2)
        integer(c_int32_t), intent(inout) :: edges(4, ntris)
        integer, intent(in) :: nedges
        ! Outputs
        integer, intent(out) :: err_code
        ! Local variables
        integer(c_int32_t) :: changed_edges(4, 4)
        integer :: iedge
        integer(c_int32_t) :: vk(2), vl(2), vm(2), vn(2)
        integer :: k, l, m, n
        integer :: jside
        integer :: itri, iside, jtri
        logical :: swapped

        err_code = ERR_NO_ERROR

        do
            swapped = .false.
            do iedge = 1, nedges
                if (.not. edge_match(faces, edges(:, iedge))) then
                    err_code = ERR_COMPUTATION_FAILURE
                    return
                end if
                itri = edges(1, iedge)
                iside = edges(2, iedge)
                k = faces(modshift(iside, 1, 3), itri)
                l = faces(modshift(iside, 2, 3), itri)
                ! Skip if this is the constraint
                if ((k == edge(1) .and. l == edge(2)) .or. &
                    (k == edge(2) .and. l == edge(1))) cycle
                vk = vtxs(:, k)
                vl = vtxs(:, l)
                ! Note: 'nabrs' ordered such that the i-th edge is
                ! composed of the j-th and k-th vertices of the facet
                m = faces(iside, itri)
                vm = vtxs(:, m)
                call find_edge_sharing_triangle( &
                    nabrs, itri, iside, jtri, jside, err_code)
                if (err_code /= ERR_NO_ERROR) return
                n = faces(jside, jtri)
                vn = vtxs(:, n)

                if (.not. is_convex(vm, vn, vk, vl)) cycle
                if (.not. incircle(vk, vl, vm, vn) > 0) cycle
                call flip_triangle_edge( &
                    vtxs, faces, nabrs, nvtxs, ntris, itri, iside, &
                    changed_edges, err_code)
                if (err_code /= ERR_NO_ERROR) return
                swapped = .true.
                ! Update changed edges in the record (including itself)
                call update_edge_record(edges, nedges, changed_edges)
                edges(:, iedge) = [itri, 2, min(m, n), max(m, n)]
            end do
            if (.not. swapped) exit
        end do
    end subroutine restore_deluanay_triangulation

    !> Recovers one constraint as a mesh edge using iterative flips.
    !!
    !! Existing mesh edges in 'locked_edges' are never flipped.
    !! After the constraint is recovered, eligible new edges are
    !! flipped to restore the local Delaunay condition without
    !! removing it.
    pure subroutine recover_constraint_edge( &
        vtxs, nvtxs, faces, nabrs, ntris, &
        edge, locked_edges, nledges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the mesh.
        integer, intent(in) :: ntris
            !! Number of triangles in the mesh.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2-D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: faces(3, ntris)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(inout) :: nabrs(3, ntris)
            !! 1-based triangle neighbours, updated in place.
        integer(c_int32_t), intent(in) :: edge(2)
            !! 1-based endpoint vertex IDs of the constraint.
        integer, intent(in) :: nledges
            !! Number of locked constraint edges.
        integer(c_int32_t), intent(in) :: locked_edges(2, nledges)
            !! 1-based endpoint pairs that must be preserved.
        ! Outputs
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 2: crossing-edge workspace allocation failed
            !! - 4: no legal sequence of flips recovers the
            !!     constraint
        ! Local variables
        integer(c_int32_t) :: xngs(4, ntris)
            !! Descriptor columns [itri, iside, vtx1, vtx2] for
            !! crossing interior mesh edges.
        integer(c_int32_t) :: new_edges(4, ntris)
        integer :: nedges
        integer :: ixng, nxngs
        integer :: nfailed

        err_code = ERR_NO_ERROR

        ! Check if constraint already satisfied
        if (any(any(faces == edge(1), 1) .and. &
                any(faces == edge(2), 1))) then
            return
        end if

        ! Find intersecting edges
        call find_crossing_edges( &
            vtxs, faces, nabrs, nvtxs, ntris, edge, xngs, nxngs, &
            err_code)
        if (err_code /= ERR_NO_ERROR) return
        if (nxngs <= 0) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if
        ! Check the edges are actually flippable
        do ixng = 1, nxngs
            if (edge_locked( &
                xngs(3:4, ixng), locked_edges, nledges)) then
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do

        ! Loop through the crossing edges
        ixng = 0
        nedges = 0
        nfailed = 0
        do while (nxngs > 0)
            ixng = modshift(ixng, 1, nxngs)
            ! Make sure the crossing data is not corrupted
            if (.not. edge_match(faces, xngs(:, ixng))) then
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
            call remove_crossing( &
                vtxs, faces, nabrs, nvtxs, ntris, edge, &
                xngs, nxngs, ixng, new_edges, nedges, nfailed, &
                err_code)
            if (err_code /= ERR_NO_ERROR) return
            ! Exit if the constraint is recovered
            if (new_edges(3, nedges) == minval(edge) .and. &
                new_edges(4, nedges) == maxval(edge)) exit
        end do

        ! Make sure the constraint is successfully recovered
        if (.not. any(any(faces == edge(1), 1) .and. &
                      any(faces == edge(2), 1))) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        ! Loop through all new edges to check their Deluanay condition
        call restore_deluanay_triangulation( &
            vtxs, faces, nabrs, nvtxs, ntris, edge, new_edges, &
            nedges, err_code)
    end subroutine recover_constraint_edge

    !> Recovers non-crossing constraint edges sequentially while
    !! preserving every earlier constraint.
    !!
    !! 'faces' is updated in place. Each successfully recovered edge
    !! is passed to the next recovery step as a locked edge.
    pure subroutine recover_constraint_edges( &
        vtxs, nvtxs, faces, ntris, edges, nedges, nabrs, &
        failed_edge, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs, ntris, nedges
            !! Numbers of vertices, triangles, and constraints.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: faces(3, ntris)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(in) :: edges(2, nedges)
            !! 1-based constraint endpoint pairs in recovery order.
        ! Outputs
        integer(c_int32_t), intent(out) :: nabrs(3, ntris)
            !! 1-based triangle neighbours for the recovered mesh.
        integer, intent(out) :: failed_edge
            !! 1-based position of the failed constraint, or zero
            !! when all constraints were recovered.
        integer, intent(out) :: err_code
            !! Shared backend status code propagated from neighbour
            !! construction or single-edge recovery.
        ! Local variables
        integer :: iedge

        failed_edge = 0
        err_code = ERR_NO_ERROR

        call find_triangle_neighbours(ntris, faces, nabrs, err_code)
        if (err_code /= ERR_NO_ERROR) return

        do iedge = 1, nedges
            call recover_constraint_edge( &
                vtxs, nvtxs, faces, nabrs, ntris, edges(:, iedge), &
                edges(:, :iedge - 1), iedge - 1, err_code)
            if (err_code /= ERR_NO_ERROR) then
                failed_edge = iedge
                return
            end if
        end do
    end subroutine recover_constraint_edges
end module meshing_triangulation
