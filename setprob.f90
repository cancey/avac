

      subroutine setprob
      implicit none
      integer coulomb
      double precision snow_density, xi, mu, threshold, beta_slope
      character*12 fname
      integer iunit
      common /voellmy/ snow_density, xi, mu, threshold, beta_slope, coulomb

!
!     # read data values for this problem
!
!
      iunit = 7
      fname = 'voellmy.data'
!     # open the unit with new routine from Clawpack 4.4 to skip over
!     # comment lines starting with #:
      call opendatafile(iunit, fname)


!     # These parameters are used in qinit.f
      read(7,*) snow_density
      read(7,*) xi
      read(7,*) mu
      read(7,*) threshold
      read(7,*) beta_slope
      read(7,*) coulomb


      close(unit=7)
      print *, 'rho = ',snow_density
      print *, 'xi = ',xi
      print *, 'mu = ',mu
      print *, 'u_* = ',threshold
      print *, 'beta = ',beta_slope
      print *, 'coulomb = ',coulomb

      end subroutine setprob
