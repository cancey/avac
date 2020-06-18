subroutine src2(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux,t,dt)

    use geoclaw_module, only: g => grav, coriolis_forcing, coriolis
    use geoclaw_module, only: friction_forcing, friction_depth
    use geoclaw_module, only: manning_coefficient
    use geoclaw_module, only: manning_break, num_manning
    !use geoclaw_module, only: spherical_distance, coordinate_system
    !use geoclaw_module, only: RAD2DEG, pi, dry_tolerance
    use geoclaw_module, only: rho_air


    !use voellmy_module

    use friction_module, only: variable_friction, friction_index
    implicit none
    double precision snow_density, xi, mu, threshold, beta_slope
    common /voellmy/ snow_density, xi, mu, threshold, beta_slope



    ! Input parameters
    integer, intent(in) :: meqn,mbc,mx,my,maux
    double precision, intent(in) :: xlower,ylower,dx,dy,t,dt


    ! Output
    double precision, intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    double precision, intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)

    ! Locals
    integer :: i, j, nman, coulomb
    real(kind=8) :: h, hu, hv, gamma, dgamma, y, fdt, a(2,2), coeff
    real(kind=8) :: xm, xc, xp, ym, yc, yp, dx_meters, dy_meters
    real(kind=8) :: u, v, hu0, hv0, delta

    real(kind=8) :: Ddt, sloc(2)
    real(kind=8) :: xiv, muv, thetax, thetay, sgnx, sgny, thetac, ucrit, rho

    ! Algorithm parameters
    ! Parameter controls when to zero out the momentum at a depth in the
    ! friction source term
    real(kind=8), parameter :: depth_tolerance = 1.0d-30

    ! Physics
    ! Nominal density of water
    !real(kind=8) :: snow_density, xi, mu, threshold, rho
    ! real(kind=8), parameter :: rho = snow_density
    !call set_voellmy('voellmy.data')
    rho = snow_density

    ! Friction source term
    if (friction_forcing) then
        do j=1,my
            do i=1,mx
                ! Extract appropriate momentum
                if (q(1,i,j) < depth_tolerance) then
                    q(2:3,i,j) = 0.d0
                else
                    ! Apply friction source term only if in shallower water
                    if (q(1,i,j) <= friction_depth) then
                        if (.not.variable_friction) then
                            do nman = num_manning, 1, -1
                                if (aux(1,i,j) .lt. manning_break(nman)) then
                                    coeff = manning_coefficient(nman)
                                endif
                            enddo
                        else
                            coeff = aux(friction_index,i,j)
                        endif
                        !  Voellmy parameter

                        xiv = xi
                        muv = mu
                        ucrit = threshold
                        thetax = atan((aux(1,i,j)-aux(1,i+1,j))/dx)
                        thetay = atan((aux(1,i,j)-aux(1,i,j+1))/dy)
                        thetac = beta_slope*sqrt(thetax**2+thetay**2)
                        ! Calculate source term
                        if (q(1, i, j) >0.d0) then
                             gamma = sqrt(q(2,i,j)**2 + q(3,i,j)**2) * g     &
                              /xiv / (q(1,i,j)**2)
                        else
                             gamma = 0.d0
                        endif
                        if ((q(2, i, j) /=0.) .and. (q(3, i, j)/=0.)) then
                                  delta = muv*g*q(1,i,j)/               &
                                  sqrt(q(2,i,j)**2 + q(3,i,j)**2)
                        else
                                  delta = 0.d0
                        endif


                        !if (q(2, i, j)<0.) then
                         !  sgnx = -1.
                         !  else if (q(2, i, j)==0.) then
                         !  sgnx= 0.
                         !  else if (q(2, i, j)>0.) then
                          ! sgnx = 1.
                        !endif

                       !if (q(3, i, j)<0.) then
                         !  sgny = -1.
                         !  else if (q(3, i, j)==0.) then
                         !  sgny= 0.
                         !  else if (q(3, i, j)>0.) then
                         !  sgny = 1.
                        !endif

                        if (coulomb==1) then
                        dgamma = 1.d0
                        else
                        dgamma = 1.d0 + dt * gamma
                        endif
                        
                        q(2, i, j) = q(2, i, j)/(dgamma + dt*delta)
                        q(3, i, j) = q(3, i, j)/(dgamma + dt*delta)
                        if ((thetac<muv) .and. (abs(q(2, i, j))<         &
                        ucrit*q(1, i, j)) .and. (abs(q(3, i, j))<        &
                        ucrit*q(1, i, j))) then
                          q(2, i, j)=0.
                          q(3, i, j)=0.
                        endif
                    endif
                endif
            enddo
        enddo
    endif
    ! End of friction source term



end subroutine src2
