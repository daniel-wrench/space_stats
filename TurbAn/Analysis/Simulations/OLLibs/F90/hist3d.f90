subroutine hist3d(vx,vy,vz,minmax,lenv,nx,ny,nz,dfn)
   implicit none
   integer, intent(in) :: lenv, nx, ny, nz
   double precision, intent(in), dimension(lenv) :: vx, vy, vz
   double precision, intent(in), optional, dimension(6) :: minmax
   integer*8, intent(out), dimension(nx,ny,nz) :: dfn
   integer :: i,j,k,l,clip_int
   double precision :: dvx, dvy, dvz, mm(6)

   if (present(minmax)) then
      write(*,*) 'using ',mm,' for velocity extrema'
      mm = minmax
   else 
      write(*,*) 'using the min-max of particle array for velocity extrema'
      mm=(/ minval(vx) , maxval(vx) &
           ,minval(vy) , maxval(vy) &
           ,minval(vz) , maxval(vz) /)
   endif

   dvx = (mm(2) - mm(1))/nx
   dvy = (mm(4) - mm(3))/ny
   dvz = (mm(6) - mm(5))/nz

   write(*,*) dvx,dvy,dvz

   do i=1,lenv
      if ( isnan(vx(i)) .or. isnan(vy(i)) &
      .or. isnan(vz(i)) ) then
         continue
      else
         j=clip_int(floor((vx(i)-mm(1))/dvx + 0.5),1,nx)
         k=clip_int(floor((vy(i)-mm(3))/dvy + 0.5),1,ny)
         l=clip_int(floor((vz(i)-mm(5))/dvz + 0.5),1,nz)
         
         dfn(j,k,l) = dfn(j,k,l) + 1
      endif
   enddo
end subroutine hist3d

double precision function clip_dp(v,minv,maxv)
   double precision, intent(in) :: v, minv, maxv
   clip_dp = minval((/maxval((/v,minv/)),maxv/))
end function
integer function clip_int(v,minv,maxv)
   integer, intent(in) :: v, minv, maxv
   clip_int = minval((/maxval((/v,minv/)),maxv/))
end function
