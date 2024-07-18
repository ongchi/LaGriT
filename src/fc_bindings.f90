pure function f_str(c_str)
   use iso_c_binding
   implicit none

   character(len=1, kind=c_char), intent(in) :: c_str(*)
   character(len=256) :: f_str
   integer :: i

   f_str = ""

   do i = 1, 256
      if (c_str(i) == c_null_char) then
         exit
      end if
      f_str(i:i) = c_str(i)
   end do

   return
end function f_str

subroutine write_c_str(f_str, c_str)

   use iso_c_binding
   implicit none

   character(*) ::f_str
   character(len=1, kind=c_char), intent(inout) :: c_str(*)

   integer :: i, n

   n = len_trim(f_str)
   do i = 1, n
      c_str(i) = f_str(i:i)
   end do
   c_str(n + 1) = c_null_char

end subroutine write_c_str

subroutine fc_cmo_get_mesh_type(cmo_name, mesh_type, imesh_type, err_no) bind(c)

   use iso_c_binding
   implicit none

   character(len=1, kind=c_char), intent(in) :: cmo_name(*)
   character(len=1, kind=c_char), intent(inout) :: mesh_type(*)
   integer(c_long), intent(inout) :: imesh_type, err_no

   integer :: i, n
   character(len=32) :: buff

   ! general function
   integer :: icharlnf
   character(len=256) :: f_str

   call cmo_get_mesh_type( &
      f_str(cmo_name), buff, imesh_type, err_no)

   call write_c_str(buff, mesh_type)

end subroutine fc_cmo_get_mesh_type
