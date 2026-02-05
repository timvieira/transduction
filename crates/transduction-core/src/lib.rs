pub mod fst;
pub mod precover;
pub mod powerset;
pub mod decompose;
pub mod token_decompose;
pub mod peekaboo;
pub mod py;

use pyo3::prelude::*;

#[pymodule]
fn transduction_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::RustFst>()?;
    m.add_class::<py::RustFsa>()?;
    m.add_class::<py::RustProfileStats>()?;
    m.add_class::<py::DecompResult>()?;
    m.add_class::<py::PeekabooDecompResult>()?;
    m.add_function(wrap_pyfunction!(py::rust_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(py::rust_peekaboo, m)?)?;
    Ok(())
}
