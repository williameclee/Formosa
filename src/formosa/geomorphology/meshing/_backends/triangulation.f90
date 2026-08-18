!> Unconstrained triangulation using the Fortran backend.
!!
!! This internal module implements incremental Bowyer-Watson
!! triangulation and facet-neighbour construction for integer 2D
!! coordinates. Constrained edge recovery is implemented separately
!! in 'constrained_triangulation.f90'. Internal coordinates use 64-
!! bit integers so the super-triangle can enclose any int32 input.
!! Triangle and vertex IDs are 1-based internally; the Python
!! wrapper returns 0-based IDs.
!!
!! Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
!! Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)

module meshing_triangulation
    use iso_c_binding, only: c_int32_t, c_int64_t
    use utils, only: ERR_NO_ERROR, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW, &
                     ERR_COMPUTATION_FAILURE
    use utils, only: modshift
    use intersections, only: incircle_pos_int32, orient
    private :: make_initial_facets, insert_vertex, toggle_edge
    private :: is_bad_finite_facet, is_bad_infinite_facet, &
               is_bad_facet
    private :: find_facet_side_neighbour
    ! Moule variables
    integer(c_int32_t), parameter :: no_nabr = -1
contains
    pure subroutine make_initial_facets( &
        vtxs, faces, seeds, iinf, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! 2D index coordinates of the vertices.
        ! Outputs
        integer(c_int32_t), intent(out) :: faces(:, :)
        integer(c_int32_t), intent(out) :: seeds(3)
        integer(c_int32_t), intent(out) :: iinf
        integer, intent(out) :: err_code
        ! Local variables
        integer(c_int32_t) :: ivtx, jvtx
        integer(c_int32_t) :: o
        integer(c_int32_t) :: face(3)

        err_code = ERR_NO_ERROR

        ! Find the first facet
        ivtx = 0
        do jvtx = 3, size(vtxs, 2)
            o = orient(vtxs(:, 1), vtxs(:, 2), vtxs(:, jvtx))
            if (o /= 0) then
                ivtx = jvtx
                exit
            end if
        end do

        if (ivtx == 0) then
            ! All vertices are collinear
            err_code = ERR_COMPUTATION_FAILURE
            return
        elseif (size(faces, 2) < 4) then
            ! Not enough space to store all the triangles
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        seeds = [1, 2, ivtx]
        iinf = size(vtxs, 2) + 1

        if (o > 0) then
            face = [1, 2, ivtx]
        else
            face = [1, ivtx, 2]
        end if
        faces(:, 1) = face
        faces(:, 2) = [iinf, face(2), face(1)]
        faces(:, 3) = [iinf, face(3), face(2)]
        faces(:, 4) = [iinf, face(1), face(3)]
    end subroutine make_initial_facets

    pure logical function is_bad_finite_facet(face, ivtx, vtxs) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
        integer(c_int32_t), intent(in) :: face(3)
            !! Triangle vertex IDs in counterclockwise order.
        integer(c_int32_t), intent(in) :: ivtx
            !! ID of the candidate vertex being inserted.
        flag = incircle_pos_int32( &
               vtxs(:, face(1)), vtxs(:, face(2)), vtxs(:, face(3)), &
               vtxs(:, ivtx))
    end function is_bad_finite_facet

    pure logical function is_bad_infinite_facet(face, ivtx, vtxs, iinf) result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
        integer(c_int32_t), intent(in) :: face(3)
            !! Triangle vertex IDs in counterclockwise order.
        integer(c_int32_t), intent(in) :: ivtx
            !! ID of the candidate vertex being inserted.
        integer(c_int32_t), intent(in) :: iinf
            !! Symbolic infinite-vertex ID.
        ! Local variables
        integer :: jinf, jvtx, kvtx
        integer :: o

        if (face(1) == iinf) then
            jinf = 1
        elseif (face(2) == iinf) then
            jinf = 2
        else
            jinf = 3
        end if
        jvtx = face(modshift(jinf, 1, 3))
        kvtx = face(modshift(jinf, 2, 3))

        o = orient(vtxs(:, jvtx), vtxs(:, kvtx), vtxs(:, ivtx))
        if (o > 0) then
            flag = .true.
        elseif (o == 0) then
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
    end function is_bad_infinite_facet

    !> Tests whether a triangle belongs to an insertion cavity.
    !!
    !! Finite triangles use the in-circle predicate. Facets
    !! containing the symbolic infinite vertex use hull visibility
    !! instead.
    pure logical function is_bad_facet(face, ivtx, vtxs, iinf) &
        result(flag)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
        integer(c_int32_t), intent(in) :: face(3)
            !! Triangle vertex IDs in counterclockwise order.
        integer(c_int32_t), intent(in) :: ivtx
            !! ID of the candidate vertex being inserted.
        integer(c_int32_t), intent(in) :: iinf
            !! Symbolic infinite-vertex ID.

        ! Base case: no infinite vertex
        if (.not. any(face == iinf)) then
            flag = is_bad_finite_facet(face, ivtx, vtxs)
        else
            ! Special case: has infinite vertex
            ! Find where the infinite vertex is, and rotate the
            ! triangle
            flag = is_bad_infinite_facet(face, ivtx, vtxs, iinf)
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

    !> Inserts 1 vertex using the Bowyer-Watson cavity operation.
    !! Triangles whose circumcircles contain the vertex are replaced
    !! by counterclockwise triangles joining the vertex to the
    !! cavity boundary.
    subroutine insert_vertex( &
        ivtx, vtxs, faces, nfaces, iinf, &
        bad_faces, bad_mask, edges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: ivtx
            !! ID of the vertex in 'vtxs' to insert.
        integer(c_int32_t), intent(in) :: vtxs(:, :)
            !! Coordinates of the vertices.
            !! This should include the vertices of the super-
            !! triangle.
        integer(c_int32_t), intent(in) :: iinf
        integer(c_int32_t), intent(inout) :: faces(:, :)
            !! Vertex indices of the triangles
        integer, intent(inout) :: nfaces
            !! Number of triangles actually in the 'triangles'
            !! array.
        integer, intent(out) :: bad_faces(size(faces, 2))
            !! Workspace containing IDs of triangles in the cavity.
        logical, intent(out) :: bad_mask(size(faces, 2))
            !! Workspace containing IDs of triangles in the cavity.
        integer, intent(inout) :: edges(2, size(vtxs, 2))
            !! Workspace containing canonical cavity-boundary edges.
        integer, intent(inout) :: err_code
            !! Shared backend status code.
        ! Local variables
        integer :: iface, ibadface, nbadfaces
        integer :: nedges, iedge
        integer :: jvtx, kvtx
            !! Endpoint IDs of the current cavity edge.
        integer(c_int64_t) :: o
            !! Signed orientation determinant for a candidate
            !! triangle.

        err_code = ERR_NO_ERROR

        ! Find triangles whose circumcircle contains the new vertex
        !$omp PARALLEL DO DEFAULT(SHARED) PRIVATE(iface) &
        !$omp SCHEDULE(STATIC)
        do iface = 1, nfaces
            bad_mask(iface) = is_bad_facet(faces(:, iface), ivtx, vtxs, iinf)
        end do
        !$omp END PARALLEL DO

        nbadfaces = 0
        do iface = 1, nfaces
            if (.not. bad_mask(iface)) cycle
            nbadfaces = nbadfaces + 1
            bad_faces(nbadfaces) = iface
        end do
        if (nbadfaces <= 0) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        ! Add bad edges to the buffer
        nedges = 0
        do ibadface = 1, nbadfaces
            iface = bad_faces(ibadface)
            call toggle_edge( &
                faces(1, iface), faces(2, iface), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
            call toggle_edge( &
                faces(2, iface), faces(3, iface), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
            call toggle_edge( &
                faces(3, iface), faces(1, iface), &
                edges, nedges, err_code)
            if (err_code /= ERR_NO_ERROR) return
        end do

        ! Make triangles from the boundary edges
        do iedge = 1, nedges
            jvtx = edges(1, iedge)
            kvtx = edges(2, iedge)

            ! Find where to insert the new triangle
            if (iedge <= nbadfaces) then
                iface = bad_faces(iedge)
            else
                if (nfaces >= size(faces, dim=2)) then
                    err_code = ERR_OVERFLOW
                    return
                end if
                nfaces = nfaces + 1
                iface = nfaces
            end if
            ! Insert CCW triangle
            o = orient( &
                vtxs(:, jvtx), vtxs(:, kvtx), vtxs(:, ivtx))

            ! Should already be oriented CCW?
            if ((count([jvtx, kvtx, ivtx] == iinf) >= 1) .or. (o > 0)) then
                faces(:, iface) = [jvtx, kvtx, ivtx]
            else
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do
    end subroutine insert_vertex

    !> Triangulates unique integer vertices using incremental
    !! Bowyer-Watson.
    !!
    !! The returned triangles use counterclockwise, 1-based vertex
    !! IDs. Their column order is not canonicalised by this routine.
    subroutine triangulate_points( &
        nvtxs, vtxs, faces, nfaces, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the input.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! Unique 2-D integer coordinates of the vertices.
        ! Outputs
        integer(c_int32_t), intent(out) :: faces(3, nvtxs*2 + 16)
            !! Counterclockwise, 1-based triangle vertex IDs.
        integer, intent(out) :: nfaces
            !! Number of active columns in 'faces'.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 1: invalid input
            !! - 2: workspace allocation failed
            !! - 3: triangle or edge capacity exceeded
            !! - 4: invalid or degenerate triangulation data
        ! Local variables
        integer :: alloc_stat
        integer :: ivtx, iface
        integer(c_int32_t) :: seeds(3)
        integer(c_int32_t) :: iinf
        integer, allocatable :: bad_faces(:)
            !! IDs of triangles in the current insertion cavity.
        logical, allocatable :: bad_mask(:)
        integer, allocatable :: edges(:, :)
            !! Canonical cavity-boundary edge workspace.

        err_code = ERR_NO_ERROR

        allocate (bad_faces(size(faces, 2)), bad_mask(size(faces, 2)), &
                  edges(2, size(vtxs, 2)), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Make the first triangles
        call make_initial_facets(vtxs, faces, seeds, iinf, err_code)
        if (err_code /= ERR_NO_ERROR) return
        nfaces = 4

        ! Insert vertices one at a time
        do ivtx = 1, nvtxs
            ! Skip already-processed seed vertices
            if (any(seeds == ivtx)) cycle
            call insert_vertex( &
                ivtx, vtxs, faces, nfaces, iinf, &
                bad_faces, bad_mask, edges, err_code)
            if (err_code /= ERR_NO_ERROR) return
        end do

        ! Remove unneeded triangles connected to the infinite vertex
        iface = 1
        do while (iface <= nfaces)
            if (any(faces(:, iface) == iinf)) then
                faces(:, iface) = faces(:, nfaces)
                nfaces = nfaces - 1
            else
                iface = iface + 1
            end if
        end do

        if (nfaces <= 0) err_code = ERR_COMPUTATION_FAILURE
    end subroutine triangulate_points

    pure subroutine find_facet_side_neighbour( &
        ivtx, jvtx, iface, iside, nabrs, edges, nedges, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: ivtx, jvtx, iface
        integer, intent(in) :: iside
        integer(c_int32_t), intent(inout) :: nabrs(:, :)
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
            nabrs(iside, iface) = edges(3, iedge)
            nabrs(edges(4, iedge), edges(3, iedge)) = iface
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
        edges(3, nedges) = iface
        edges(4, nedges) = iside
    end subroutine find_facet_side_neighbour

    !> Finds the adjacent facet across each triangle side.
    !!
    !! Side 'i' is opposite vertex 'i'. Boundary sides receive the
    !! sentinel value 'no_nabr'.
    pure subroutine find_facet_neighbours( &
        faces, nabrs, nfaces, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nfaces
            !! Number of triangles in the mesh.
        integer(c_int32_t), intent(in) :: faces(3, nfaces)
            !! 1-based facet vertex IDs.
        ! Outputs
        integer(c_int32_t), intent(out) :: nabrs(3, nfaces)
            !! 1-based adjacent facet IDs across corresponding
            !! sides, or 'no_nabr' at the mesh boundary.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 2: edge-workspace allocation failed
            !! - 3: edge-workspace capacity exceeded
        ! Local variables
        integer :: iface
        integer :: ivtx, jvtx
        integer :: nedges
        integer(c_int32_t), allocatable :: edges(:, :)
        integer :: alloc_stat
        integer :: iside

        err_code = ERR_NO_ERROR
        allocate (edges(4, nfaces*3), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        nabrs = no_nabr
        nedges = 0
        do iface = 1, nfaces
            do iside = 1, 3
                ! Skip if complement already found
                if (nabrs(iside, iface) /= no_nabr) cycle

                ivtx = faces(modulo(iside, 3) + 1, iface)
                jvtx = faces(modulo(iside + 1, 3) + 1, iface)
                call find_facet_side_neighbour( &
                    min(ivtx, jvtx), max(ivtx, jvtx), iface, iside, &
                    nabrs, edges, nedges, err_code)
                if (err_code /= ERR_NO_ERROR) return
            end do
        end do
    end subroutine find_facet_neighbours
end module meshing_triangulation
