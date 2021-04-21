#include <array>

using namespace std;
typedef uint8_t voxel_type;

// TODO: OpenACC
array<double,3> center_of_mass(const voxel_type *voxels, const array<size_t,3> &shape) {
  double cm[3] = {0,0,0};
  size_t Nz = shape[0], Ny = shape[1], Nx = shape[2];
  size_t image_length = Nx*Ny*Nz;
  
  //#pragma omp parallel for reduction(+:cm[:3])
  #pragma omp simd  
  for(uint64_t flat_idx=0;flat_idx<image_length;flat_idx++){
    uint64_t x = flat_idx % Nx;
    uint64_t y = (flat_idx / Nx) % Ny;
    uint64_t z = flat_idx / (Nx*Ny);

    double m = voxels[flat_idx];
		      
    cm[0] += m*x; cm[1] += m*y; cm[2] += m*z;
  }

  return array<double,3>{cm[0],cm[1],cm[2]};
}

array<double,9> inertia_matrix(const voxel_type *voxels, const array<size_t,3> &shape, const array<double,3> &cm)
{
  array<double,9> M = {0,0,0,
		       0,0,0,
		       0,0,0};

  size_t Nz = shape[0], Ny = shape[1], Nx = shape[2];
  size_t image_length = Nx*Ny*Nz;
  
  //#pragma omp parallel for reduction(+:M[:9])
  #pragma omp simd
  for(size_t flat_idx=0;flat_idx<image_length;flat_idx++)
    if(voxels[flat_idx] != 0) { // TODO: Check if faster with or without test
      // x,y,z
      uint64_t xs[3] = {flat_idx % Nx, (flat_idx / Nx) % Ny, flat_idx / (Nx*Ny)};
      
      for(int i=0;i<3;i++){
	M[i*3 + i] += voxels[flat_idx] * (xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);
	for(int j=0;j<3;j++)
	  M[i*3 + j] -= voxels[flat_idx] * xs[i] * xs[j];
      }
    }

  return M;
}


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace python_api { 
  namespace py = pybind11;
  
  array<double,3> center_of_mass(const py::array_t<voxel_type> &np_voxels){
    auto voxels_info    = np_voxels.request();
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    array<size_t,3> shape = {voxels_info.shape[0],voxels_info.shape[1],voxels_info.shape[2]};        

    return ::center_of_mass(voxels,shape);
  }


  array<double,9> inertia_matrix(const py::array_t<voxel_type> &np_voxels){
    auto voxels_info    = np_voxels.request();
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    array<size_t,3> shape = {voxels_info.shape[0],voxels_info.shape[1],voxels_info.shape[2]};    
    
    array<double,3> cm = ::center_of_mass(voxels, shape);
    return ::inertia_matrix(voxels,shape, cm);
  }
 
}



PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",  &python_api::center_of_mass);
    m.def("inertia_matrix",  &python_api::inertia_matrix);
}
