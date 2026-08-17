!> Recovers constrained edges using the Fortran backend.
!!
!! This internal module adds constraints to an existing
!! unconstrained triangulation by flipping crossing edges and
!! restoring the local Delaunay condition. The base triangulation
!! and facet-neighbour construction are implemented separately in
!! 'triangulation.f90'. Triangle and vertex IDs are 1-based
!! internally; the Python wrapper returns 0-based IDs.
!!
!! Created: 2026-08-17, En-Chi Lee (williameclee@gmail.com)

module meshing_cstr_triangulation
    use iso_c_binding, only: c_int32_t, c_int64_t
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, &
                     ERR_COMPUTATION_FAILURE
    use utils, only: mod1, modshift
    use intersections, only: incircle, orient, xcross, xcross_orient
    use meshing_triangulation, only: no_nabr, find_facet_neighbours
    private :: update_flipped_neighbours
    private :: edge_locked, edge_match, update_edge_record
    private :: remove_crossing, restore_delaunay_triangulation
    ! Moule variables
contains
    pure subroutine update_flipped_neighbours( &
        nabrs, iface, iside, jface, jside)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(inout) :: nabrs(:, :)
        integer, intent(in) :: iface, iside, jface, jside
        ! Local variables
        integer(c_int32_t) :: inabrs(3), jnabrs(3)
        integer :: innabr, jnnabr
        integer :: inside, inside_i, inside_j

        ! Preserve both rows and locate reciprocal outside-neighbour
        ! entries before changing the neighbour table.
        inabrs = nabrs(:, iface)
        jnabrs = nabrs(:, jface)
        innabr = jnabrs(modshift(jside, 1, 3))
        jnnabr = inabrs(modshift(iside, 1, 3))
        inside_i = 0
        inside_j = 0
        if (innabr /= no_nabr) then
            do inside = 1, 3
                if (nabrs(inside, innabr) == jface) then
                    inside_i = inside
                    exit
                end if
            end do
        end if
        if (jnnabr /= no_nabr) then
            do inside = 1, 3
                if (nabrs(inside, jnnabr) == iface) then
                    inside_j = inside
                    exit
                end if
            end do
        end if

        ! Change the two incident triangles.
        nabrs(:, iface) = &
            [jnabrs(modshift(jside, 1, 3)), jface, &
             inabrs(modshift(iside, 2, 3))]
        nabrs(:, jface) = &
            [jnabrs(modshift(jside, 2, 3)), &
             inabrs(modshift(iside, 1, 3)), iface]

        ! Update the reciprocal entries in the outside neighbours.
        if (inside_i > 0) nabrs(inside_i, innabr) = iface
        if (inside_j > 0) nabrs(inside_j, jnnabr) = jface
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
        integer(c_int64_t) :: o_abc, o_abd

        o_abc = orient(a, b, c)
        o_abd = orient(a, b, d)

        if ((o_abc == 0) .or. (o_abd == 0)) then
            ! Degenerate triangle (collinear)
            flag = .false.
            return
        elseif ((o_abc > 0) .eqv. (o_abd > 0)) then
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
    pure subroutine find_edge_sharing_facets( &
        nabrs, iface, iside, jface, jside, err_code)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: nabrs(:, :)
            !! 1-based triangle neighbours, with 'no_nabr' at
            !! the mesh boundary.
        integer, intent(in) :: iface, iside
            !! Triangle and local side identifying the input edge.
        integer, intent(out) :: jface, jside
            !! Adjacent triangle and its reciprocal local side.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 4: the edge is a boundary edge or has no
            !!     reciprocal neighbour entry

        err_code = ERR_NO_ERROR
        jface = nabrs(iside, iface)
        if (jface < lbound(nabrs, 2) .or. jface > ubound(nabrs, 2)) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if
        do jside = 1, 3
            if (nabrs(jside, jface) == iface) then
                exit
            elseif (jside == 3) then
                ! No matching side found, something must be wrong
                err_code = ERR_COMPUTATION_FAILURE
                return
            end if
        end do
    end subroutine find_edge_sharing_facets

    !> Flips an interior triangle edge in a convex quadrilateral.
    !!
    !! The routine updates the two incident triangles, their
    !! neighbour records, and reciprocal records in adjacent
    !! triangles. The new edge is side 2 of triangle 'iface' and side
    !! 3 of the other triangle.
    pure subroutine flip_quadrilateral_edge( &
        vtxs, faces, nabrs, nvtxs, nfaces, iface, iside, &
        changes, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs, nfaces
            !! Numbers of vertices and triangles in the mesh.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2-D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: faces(3, nfaces)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(inout) :: nabrs(3, nfaces)
            !! 1-based triangle neighbours, updated in place.
        integer, intent(in) :: iface, iside
            !! Triangle and local side identifying the edge to flip.
        ! Outputs
        integer(c_int32_t), intent(out) :: changes(4, 4)
            !! Descriptors '[iface, iside, vtx1, vtx2]' for the four
            !! non-flipped sides whose ownership may have changed.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 1: 'iface' or 'iside' is out of bounds
            !! - 4: the edge is on the boundary, its neighbour
            !!     record is invalid, or the quadrilateral is not
            !!     convex
        ! Local variables
        integer :: jface, jside
        integer(c_int32_t) :: j, k, l, m

        err_code = ERR_NO_ERROR

        if ((iface < 1) .or. (iface > nfaces)) then
            err_code = ERR_INVALID_INPUT
            return
        elseif ((iside < 1) .or. (iside > 3)) then
            err_code = ERR_INVALID_INPUT
            return
        end if

        ! Find the triangle/side sharing the edge
        call find_edge_sharing_facets(nabrs, iface, iside, jface, jside, err_code)
        if (err_code /= ERR_NO_ERROR) return

        ! Find the vertices
        l = faces(iside, iface)
        m = faces(jside, jface)
        j = faces(modshift(iside, 1, 3), iface)
        k = faces(modshift(iside, 2, 3), iface)

        ! Check the edge is actually flippable
        if (.not. is_convex(vtxs(:, l), vtxs(:, m), vtxs(:, j), vtxs(:, k))) then
            err_code = ERR_COMPUTATION_FAILURE
            return
        end if

        ! Flip the triangles
        faces(:, iface) = [l, j, m]
        faces(:, jface) = [l, m, k]
        ! Update the neighbours
        call update_flipped_neighbours( &
            nabrs, iface, iside, jface, jside)

        ! Record the other changed edges
        changes(:, 1) = [iface, 1, min(j, m), max(j, m)]
        changes(:, 2) = [iface, 3, min(l, j), max(l, j)]
        changes(:, 3) = [jface, 2, min(k, l), max(k, l)]
        changes(:, 4) = [jface, 1, min(m, k), max(m, k)]
    end subroutine flip_quadrilateral_edge

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
        vtxs, faces, nabrs, nvtxs, nfaces, &
        edge, xngs, nxngs, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the triangulation.
        integer, intent(in) :: nfaces
            !! Number of triangles in the triangulation.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D index coordinates of the vertices.
        integer(c_int32_t), intent(in) :: faces(3, nfaces)
            !! Vertex indices of the triangles.
        integer(c_int32_t), intent(in) :: nabrs(3, nfaces)
            !! Triangle neighbour indices across sides.
        integer, intent(in) :: edge(2)
            !! 1-based endpoint vertex IDs of the constraint edge.
        ! Outputs
        integer(c_int32_t), intent(out) :: xngs(4, nfaces)
            !! Descriptor columns [iface, iside, vtx1, vtx2] for
            !! crossing interior mesh edges.
        integer(c_int32_t), intent(out) :: nxngs
            !! Total number of crossing mesh edges found.
        integer, intent(out) :: err_code
            !! Shared backend status code:
            !! - 0: completed successfully
            !! - 2: workspace allocation failed
        ! Local variables
        integer :: ifaces(3*nfaces), isides(3*nfaces)
            !! Triangle and side IDs owning each unique interior
            !! edge.
        integer :: iface, iside, iedge, nedges, ixng
            !! Loop indices and counters for candidate and
            !! crossing edges.
        integer(c_int32_t), allocatable :: l(:), m(:)
            !! Endpoint vertex IDs of unique interior mesh edges.
        integer(c_int64_t) :: vj(2), vk(2), vjk(2)
            !! Coordinates of constraint endpoints and target
            !! constraint vector.
        integer(c_int64_t), allocatable :: vl(:, :), vm(:, :)
            !! Endpoint coordinates of unique interior mesh edges.
        integer(c_int64_t), allocatable :: &
            vlm(:, :), vlj(:, :), vlk(:, :), vjl(:, :), vjm(:, :)
            !! Difference vectors for 2D orientation calculations.
        integer(c_int64_t), allocatable :: &
            o_jkl(:), o_jkm(:), o_lmj(:), o_lmk(:)
            !! 2D cross-product orientation determinants.
        logical(kind=1), allocatable :: is_xng(:)
            !! Boolean mask identifying proper crossing edges.
        integer :: alloc_stat
            !! Dynamic allocation status code.

        err_code = ERR_NO_ERROR

        ! Extract unique interior edges
        nedges = 0
        do iedge = 1, nfaces*3
            iface = (iedge - 1)/3 + 1
            iside = mod1(iedge, 3)
            if (nabrs(iside, iface) == no_nabr) cycle
            if (iface >= nabrs(iside, iface)) cycle
            nedges = nedges + 1
            ifaces(nedges) = iface
            isides(nedges) = iside
        end do

        ! Fetch vertex coordinates and their distance vectors
        vj = vtxs(:, edge(1))
        vk = vtxs(:, edge(2))
        vjk = vk - vj
        allocate (l(nedges), m(nedges), &
                  vl(2, nedges), vm(2, nedges), vlm(2, nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        do iedge = 1, nedges
            iface = ifaces(iedge)
            iside = isides(iedge)
            l(iedge) = faces(modshift(iside, 1, 3), iface)
            m(iedge) = faces(modshift(iside, 2, 3), iface)
            vl(:, iedge) = vtxs(:, l(iedge))
            vm(:, iedge) = vtxs(:, m(iedge))
        end do

        vlm = vm - vl

        allocate (vjl(2, nedges), vjm(2, nedges), &
                  o_jkl(nedges), o_jkm(nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Compute orientation of edge endpoints relative to
        ! constraint vector
        vjl = vl - spread(vj, dim=2, ncopies=nedges)
        vjm = vm - spread(vj, dim=2, ncopies=nedges)
        o_jkl = vjk(1)*vjl(2, :) - vjk(2)*vjl(1, :)
        o_jkm = vjk(1)*vjm(2, :) - vjk(2)*vjm(1, :)

        allocate (o_lmj(nedges), o_lmk(nedges), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Compute orientation of constraint endpoints relative to
        ! mesh edge vectors
        call move_alloc(from=vjl, to=vlj)
        call move_alloc(from=vjm, to=vlk)
        vlm = vm - vl
        vlj = spread(vj, dim=2, ncopies=nedges) - vl
        vlk = spread(vk, dim=2, ncopies=nedges) - vl
        o_lmj = vlm(1, :)*vlj(2, :) - vlm(2, :)*vlj(1, :)
        o_lmk = vlm(1, :)*vlk(2, :) - vlm(2, :)*vlk(1, :)

        ! Classify proper line-segment crossings (Xs) with strict
        ! opposite orientations
        allocate (is_xng(nedges), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        is_xng = xcross_orient(o_jkl, o_jkm, o_lmj, o_lmk)
        nxngs = count(is_xng)

        ! Pack crossing edge descriptors into output matrix
        ixng = 0
        do iedge = 1, nedges
            if (.not. is_xng(iedge)) cycle
            ixng = ixng + 1
            xngs(:, ixng) = &
                [ifaces(iedge), isides(iedge), &
                 min(l(iedge), m(iedge)), max(l(iedge), m(iedge))]
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
    pure logical function edge_match(faces, edge)
        implicit none(type, external)
        ! Arguments
        integer(c_int32_t), intent(in) :: faces(:, :)
        integer, intent(in) :: edge(4)
        ! Local variables
        integer :: j, k
            !! Actual vertex indices for the iside-th edge of the
            !! iface-th triangle.

        j = faces(modshift(edge(2), 1, 3), edge(1))
        k = faces(modshift(edge(2), 2, 3), edge(1))
        edge_match = (edge(3) == j .and. edge(4) == k) .or. &
                     (edge(3) == k .and. edge(4) == j)
    end function edge_match

    pure subroutine remove_crossing( &
        vtxs, faces, nabrs, nvtxs, nfaces, edge, &
        xngs, nxngs, ixng, new_edges, nedges, nfailed, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
        integer, intent(in) :: nfaces
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
        integer(c_int32_t), intent(inout) :: faces(3, nfaces)
        integer(c_int32_t), intent(inout) :: nabrs(3, nfaces)
        integer(c_int32_t), intent(in) :: edge(2)
        integer(c_int32_t), intent(inout) :: xngs(4, nfaces)
        integer, intent(inout) :: nxngs, ixng
        integer(c_int32_t), intent(inout) :: new_edges(4, nfaces)
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
        call find_edge_sharing_facets(nabrs, iface, iside, jface, jside, err_code)
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
        call flip_quadrilateral_edge( &
            vtxs, faces, nabrs, nvtxs, nfaces, iface, iside, &
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

    pure subroutine restore_delaunay_triangulation( &
        vtxs, faces, nabrs, nvtxs, nfaces, edge, edges, nedges, &
        err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
        integer, intent(in) :: nfaces
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
        integer(c_int32_t), intent(inout) :: faces(3, nfaces)
        integer(c_int32_t), intent(inout) :: nabrs(3, nfaces)
        integer(c_int32_t), intent(in) :: edge(2)
        integer(c_int32_t), intent(inout) :: edges(4, nfaces)
        integer, intent(in) :: nedges
        ! Outputs
        integer, intent(out) :: err_code
        ! Local variables
        integer(c_int32_t) :: changed_edges(4, 4)
        integer :: iedge
        integer(c_int32_t) :: vk(2), vl(2), vm(2), vn(2)
        integer :: k, l, m, n
        integer :: jside
        integer :: iface, iside, jface
        logical :: swapped

        err_code = ERR_NO_ERROR

        do
            swapped = .false.
            do iedge = 1, nedges
                if (.not. edge_match(faces, edges(:, iedge))) then
                    err_code = ERR_COMPUTATION_FAILURE
                    return
                end if
                iface = edges(1, iedge)
                iside = edges(2, iedge)
                k = faces(modshift(iside, 1, 3), iface)
                l = faces(modshift(iside, 2, 3), iface)
                ! Skip if this is the constraint
                if ((k == edge(1) .and. l == edge(2)) .or. &
                    (k == edge(2) .and. l == edge(1))) cycle
                vk = vtxs(:, k)
                vl = vtxs(:, l)
                ! Note: 'nabrs' ordered such that the i-th edge is
                ! composed of the j-th and k-th vertices of the facet
                m = faces(iside, iface)
                vm = vtxs(:, m)
                call find_edge_sharing_facets( &
                    nabrs, iface, iside, jface, jside, err_code)
                if (err_code /= ERR_NO_ERROR) return
                n = faces(jside, jface)
                vn = vtxs(:, n)

                if (.not. is_convex(vm, vn, vk, vl)) cycle
                if (.not. incircle(vk, vl, vm, vn) > 0) cycle
                call flip_quadrilateral_edge( &
                    vtxs, faces, nabrs, nvtxs, nfaces, iface, iside, &
                    changed_edges, err_code)
                if (err_code /= ERR_NO_ERROR) return
                swapped = .true.
                ! Update changed edges in the record (including itself)
                call update_edge_record(edges, nedges, changed_edges)
                edges(:, iedge) = [iface, 2, min(m, n), max(m, n)]
            end do
            if (.not. swapped) exit
        end do
    end subroutine restore_delaunay_triangulation

    !> Recovers a constraint as a mesh edge using iterative flips.
    !!
    !! Existing mesh edges in 'locked_edges' are never flipped.
    !! After the constraint is recovered, eligible new edges are
    !! flipped to restore the local Delaunay condition without
    !! removing it.
    pure subroutine recover_constraint_edge( &
        vtxs, nvtxs, faces, nabrs, nfaces, &
        edge, locked_edges, nledges, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs
            !! Number of vertices in the mesh.
        integer, intent(in) :: nfaces
            !! Number of triangles in the mesh.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2-D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: faces(3, nfaces)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(inout) :: nabrs(3, nfaces)
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
        integer(c_int32_t) :: xngs(4, nfaces)
            !! Descriptor columns [iface, iside, vtx1, vtx2] for
            !! crossing interior mesh edges.
        integer(c_int32_t) :: new_edges(4, nfaces)
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
            vtxs, faces, nabrs, nvtxs, nfaces, edge, xngs, nxngs, &
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
                vtxs, faces, nabrs, nvtxs, nfaces, edge, &
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

        ! Loop through all new edges to check their Delaunay condition
        call restore_delaunay_triangulation( &
            vtxs, faces, nabrs, nvtxs, nfaces, edge, new_edges, &
            nedges, err_code)
    end subroutine recover_constraint_edge

    !> Recovers non-crossing constraint edges sequentially while
    !! preserving every earlier constraint.
    !!
    !! 'faces' is updated in place. Each successfully recovered edge
    !! is passed to the next recovery step as a locked edge.
    pure subroutine recover_constraint_edges( &
        vtxs, faces, nvtxs, nfaces, edges, nedges, nabrs, &
        failed_edge, err_code)
        implicit none(type, external)
        ! Arguments
        integer, intent(in) :: nvtxs, nfaces, nedges
            !! Numbers of vertices, triangles, and constraints.
        integer(c_int32_t), intent(in) :: vtxs(2, nvtxs)
            !! 2D integer vertex coordinates.
        integer(c_int32_t), intent(inout) :: faces(3, nfaces)
            !! 1-based triangle vertex IDs, updated in place.
        integer(c_int32_t), intent(in) :: edges(2, nedges)
            !! 1-based constraint endpoint pairs in recovery order.
        ! Outputs
        integer(c_int32_t), intent(out) :: nabrs(3, nfaces)
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

        call find_facet_neighbours(faces, nabrs, nfaces, err_code)
        if (err_code /= ERR_NO_ERROR) return

        do iedge = 1, nedges
            call recover_constraint_edge( &
                vtxs, nvtxs, faces, nabrs, nfaces, edges(:, iedge), &
                edges(:, :iedge - 1), iedge - 1, err_code)
            if (err_code /= ERR_NO_ERROR) then
                failed_edge = iedge
                return
            end if
        end do
    end subroutine recover_constraint_edges
end module meshing_cstr_triangulation
