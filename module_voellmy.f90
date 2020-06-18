module voellmy_module
    implicit none
    save
    real(kind=8) ::  snow_density, xi, mu, threshold, beta_slope
    character(len=*), parameter  ::  fname = 'voellmy.data'
    integer, parameter :: iunit = 7, ioutput = 70
    integer :: coulomb


    contains

    subroutine set_voellmy(file_name)
    character(len=*), intent(in), optional :: file_name

    call opendatafile(iunit, fname)



    read(iunit,*) snow_density
    read(iunit,*) xi
    read(iunit,*) mu
    read(iunit,*) threshold
    read(iunit,*) beta_slope
    read(iunit,*) beta_slope
    read(iunit,*) coulomb
    close(unit=iunit)
    open(unit=ioutput,file='source.data',status="unknown",action="write")

    write(ioutput,*) ' '
    write(ioutput,*) '--------------------------------------------'
    write(ioutput,*) 'Voellmy was used with:'
    write(ioutput,*)  '-------------------'

    write(ioutput,*) 'rho = ',snow_density
    write(ioutput,*) 'xi = ',xi
    write(ioutput,*) 'mu = ',mu
    write(ioutput,*) 'u_* = ',threshold
    write(ioutput,*) 'beta = ',beta_slope
    if (coulomb == 0) then
    write(ioutput,*) 'model = voellmy'
    else
    write(ioutput,*) 'model = coulomb'
    endif
    close(unit=ioutput)
    end subroutine set_voellmy
end module voellmy_module
