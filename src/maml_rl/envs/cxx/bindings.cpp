#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "MarsLander.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mars_lander_cpp, m) {
    py::class_<MarsLander>(m, "MarsLander")
        .def(py::init<>())
        .def("reset", &MarsLander::reset)
        .def("step", &MarsLander::step)
        .def("get_obs", &MarsLander::get_obs);
}
