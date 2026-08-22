module util_structs
    use utils, only: ERR_NO_ERROR, ERR_INVALID_INPUT, &
                     ERR_ALLOCATION_FAILURE, ERR_OVERFLOW
    implicit none(type, external)

	!> Stores maximum elevations for a level of a spatial pyramid.
    type :: pyramid_level
        real, allocatable :: zmax(:, :)
            !! Maximum valid elevation per block
        integer :: block_size = 1
            !! Block width in original DEM cells
    end type pyramid_level

    !> Describes the maximum-elevation pyramid for a DEM.
    type :: elevation_pyramid
        type(pyramid_level), allocatable :: levels(:)
            !! Pyramid levels
        integer :: dem_nrows
            !! Number of rows in the source DEM
        integer :: dem_ncols
            !! Number of columns in the source DEM
        real :: invalid_value
            !! Sentinel used for invalid leaf cells
        integer :: factor
            !! Reduction factor between adjacent levels
    end type elevation_pyramid
contains
    !> Builds a maximum-elevation pyramid from a DEM and validity
    !! mask.
    !!
    !! Level 0 contains individual DEM cells. Each higher level
    !! stores the maximum elevation in a 'fac' by 'fac' group of
    !! child blocks. Invalid leaf cells receive a negative sentinel
    !! so they cannot qualify as isolation limit points.
    pure subroutine build_elevation_pyramid( &
        z, valids, p, fac, err_code)
        implicit none(type, external)
        ! Arguments
        real, intent(in) :: z(:, :)
            !! Elevation grid.
        logical(kind=1), intent(in) :: valids(:, :)
            !! Validity mask
        integer, intent(in) :: fac
            !! Reduction factor between levels
        type(elevation_pyramid), intent(out) :: p
            !! Constructed pyramid
        integer, intent(out) :: err_code
            !! Backend status code
        ! Local variables
        integer :: nrows, ncols
        integer :: cif, cil, cjf, cjl, pi, pj
        integer :: ilvl, nlvls
        integer :: alloc_stat

        err_code = ERR_NO_ERROR
        if (fac < 2) then
            err_code = ERR_INVALID_INPUT
            return
        elseif (any(shape(z) /= shape(valids))) then
            err_code = ERR_INVALID_INPUT
            return
        end if

        ! Count number of levels required
        nrows = size(z, dim=1)
        ncols = size(z, dim=2)
        nlvls = 0
        do while (nrows > 1 .or. ncols > 1)
            nlvls = nlvls + 1
            nrows = (nrows + fac - 1)/fac
            ncols = (ncols + fac - 1)/fac
        end do

        allocate (p%levels(0:nlvls), stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if

        ! Initialise pyramid
        p%factor = fac
        p%invalid_value = -huge(0.)
        p%levels(0)%block_size = 1
        p%dem_nrows = size(z, dim=1)
        p%dem_ncols = size(z, dim=2)
        allocate (p%levels(0)%zmax(p%dem_nrows, p%dem_ncols), &
                  stat=alloc_stat)
        if (alloc_stat /= 0) then
            err_code = ERR_ALLOCATION_FAILURE
            return
        end if
        p%levels(0)%zmax = z
        where (.not. valids)
            p%levels(0)%zmax = p%invalid_value
        end where

        ! Build parent level
        do ilvl = 1, nlvls
            p%levels(ilvl)%block_size = &
                p%levels(ilvl - 1)%block_size*fac
            allocate (p%levels(ilvl)%zmax( &
                      (size(p%levels(ilvl - 1)%zmax, dim=1) + fac - 1)/fac, &
                      (size(p%levels(ilvl - 1)%zmax, dim=2) + fac - 1)/fac), &
                      stat=alloc_stat)
            if (alloc_stat /= 0) then
                err_code = ERR_ALLOCATION_FAILURE
                return
            end if
            ! Fill the zmax
            do pj = 1, size(p%levels(ilvl)%zmax, dim=2)
            do pi = 1, size(p%levels(ilvl)%zmax, dim=1)
                ! Find the extent of the parent
                cif = fac*(pi - 1) + 1
                cil = min(fac*pi, size(p%levels(ilvl - 1)%zmax, dim=1))
                cjf = fac*(pj - 1) + 1
                cjl = min(fac*pj, size(p%levels(ilvl - 1)%zmax, dim=2))
                p%levels(ilvl)%zmax(pi, pj) = &
                    maxval(p%levels(ilvl - 1)%zmax(cif:cil, cjf:cjl))
            end do
            end do
        end do
    end subroutine build_elevation_pyramid
end module util_structs
