pub mod fst;
pub mod precover;
pub mod powerset;
pub mod decompose;
pub mod peekaboo;
pub mod minimize;
pub mod incremental;
pub mod py;

use pyo3::prelude::*;

#[pymodule]
fn transduction_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::RustFst>()?;
    m.add_class::<py::RustFsa>()?;
    m.add_class::<py::RustProfileStats>()?;
    m.add_class::<py::DecompResult>()?;
    m.add_class::<py::RustPeekabooStats>()?;
    m.add_class::<py::PeekabooDecompResult>()?;
    m.add_class::<py::RustDirtyStateDecomp>()?;
    m.add_class::<py::DirtyStepResult>()?;
    m.add_class::<py::DirtyNextResult>()?;
    m.add_class::<py::RustDirtyPeekabooDecomp>()?;
    m.add_function(wrap_pyfunction!(py::rust_decompose, m)?)?;
    Ok(())
}
