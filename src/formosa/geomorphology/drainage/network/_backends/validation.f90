!> Validates flow-graph topology using the Fortran backend.
!!
!! This internal module is called by the Python network API and
!! other Fortran routines and is not intended to be used directly.
!!
!! Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
module network_validation
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE
    use intersections, only: lines_intersect
    implicit none(type, external)
    private :: argsort_arcs, record_topology_intersection
contains
    pure function argsort_arcs(bboxes) result(indices)
        ! Helper function for 'scan_invalid_graph_topology' to sort the arcs by the left edge of their bounding box.
        implicit none(type, external)
        ! Arguments
        real, intent(in), contiguous :: bboxes(:, :)
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

    pure subroutine record_topology_intersection(record, intxs, nintxs)
        !! Counts one detected topology violation and stores it if capacity remains.
        !!
        !! The total count is incremented even after 'intxs' is full. This lets
        !! the caller distinguish the number stored from the exact number found
        !! and retry with an exactly sized buffer when necessary.
        implicit none(type, external)
        integer, intent(in) :: record(5)
            !! Intersection record: arc IDs, segment IDs, and intersection flag
        integer, intent(inout), contiguous :: intxs(:, :)
            !! Output buffer containing up to 'size(intxs, 2)' records
        integer, intent(inout) :: nintxs
            !! Total number of violations encountered, including unstored ones

        nintxs = nintxs + 1
        if (nintxs <= size(intxs, 2)) intxs(:, nintxs) = record
    end subroutine record_topology_intersection

    !> Scans all candidate segment pairs and returns the total
    !! violation count.
    !!
    !! Only the first 'capacity' violations are stored in 'intxs'.
    subroutine scan_invalid_graph_topology( &
        vtxs, arc_endpts, capacity, intxs, nintxs, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in), contiguous :: vtxs(:, :)
            !! Vertex coordinates arranged as (2, V).
        integer, intent(in), contiguous :: arc_endpts(:, :)
            !! Inclusive, one-based arc endpoint indices arranged as '(2, narcs)'
        integer, intent(in) :: capacity
            !! Maximum number of intersection records that can be stored
        ! Outputs
        integer, intent(out) :: intxs(5, capacity)
            !! Stored intersection records; only the first
            !! 'min(nintxs, capacity)' columns are defined
        integer, intent(out) :: nintxs
            !! Exact number of violations found, which may exceed 'capacity'
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
        err_code = ERR_NO_ERROR

        if (size(arc_endpts, 1) /= 2) then
            err_code = ERR_INVALID_INPUT
            return
        else if (size(vtxs, 1) /= 2) then
            err_code = ERR_INVALID_INPUT
            return
        else if (capacity < 1) then
            err_code = ERR_INVALID_INPUT
            return
        end if
        narcs = size(arc_endpts, 2)

        if (narcs == 0) return

        ! Construct the bounding boxes for each arc
        allocate (arc_bboxes(4, narcs), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        do iarc = 1, narcs
            arc_bboxes(1, iarc) = minval(vtxs(1, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(2, iarc) = minval(vtxs(2, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(3, iarc) = maxval(vtxs(1, arc_endpts(1, iarc):arc_endpts(2, iarc)))
            arc_bboxes(4, iarc) = maxval(vtxs(2, arc_endpts(1, iarc):arc_endpts(2, iarc)))
        end do

        ! Check arcs against themselves first
        do iarc = 1, narcs
            if (arc_endpts(2, iarc) - arc_endpts(1, iarc) == 1) cycle ! Skip if arc is just a single segment
            do iseg = arc_endpts(1, iarc), arc_endpts(2, iarc) - 1
            do jseg = iseg + 1, arc_endpts(2, iarc) - 1
                intx_flag = lines_intersect( &
                            vtxs(:, iseg), vtxs(:, iseg + 1), &
                            vtxs(:, jseg), vtxs(:, jseg + 1))
                if (intx_flag > 0) then
                    call record_topology_intersection( &
                        [iarc, iarc, iseg, jseg, intx_flag], intxs, nintxs)
                end if
            end do
            end do
        end do

        allocate (idx(narcs), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        idx = argsort_arcs(arc_bboxes)

        ! Check every arc against each other
        do i = 1, narcs
            iarc = idx(i)
            jloop: do j = i + 1, narcs
                jarc = idx(j)

                ! Skip if min x of right arc is greater than max x of left arc
                if (arc_bboxes(1, jarc) > arc_bboxes(3, iarc)) exit jloop

                ! Inline fast overlap check (no min/max calls)
                if (arc_bboxes(1, iarc) > arc_bboxes(3, jarc) .or. &
                    arc_bboxes(3, iarc) < arc_bboxes(1, jarc) .or. &
                    arc_bboxes(2, iarc) > arc_bboxes(4, jarc) .or. &
                    arc_bboxes(4, iarc) < arc_bboxes(2, jarc)) cycle

                do iseg = arc_endpts(1, iarc), arc_endpts(2, iarc) - 1
                do jseg = arc_endpts(1, jarc), arc_endpts(2, jarc) - 1
                    intx_flag = lines_intersect( &
                                vtxs(:, iseg), vtxs(:, iseg + 1), &
                                vtxs(:, jseg), vtxs(:, jseg + 1))
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
            end do jloop
        end do
    end subroutine scan_invalid_graph_topology
end module network_validation
