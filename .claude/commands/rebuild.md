---
description: Build Rust crate, install wheel, and run tests
allowed-tools: [Bash, Read]
---

# Rebuild

Build the Rust transduction-core crate, install the wheel, and run the test suite.

## Steps

1. **Build** the Rust crate with maturin:
   ```
   maturin build --release -m crates/transduction-core/Cargo.toml --interpreter python3.10
   ```

2. **Install** the built wheel:
   ```
   pip install --force-reinstall crates/transduction-core/target/wheels/transduction_core-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
   ```

3. **Test** with pytest:
   ```
   python -m pytest tests/test_general.py -v
   ```

## Important Notes

- System Python is 3.10 — always use `--interpreter python3.10` (NOT conda's 3.12)
- The wheel filename includes the version; if it changes, glob for `target/wheels/*.whl`
- Expected: 90/91 pass, 1 xfail (`test_triplets_of_doom[recursive_dfa_decomp]`)

## Arguments

If the user provides arguments: $ARGUMENTS
- `--no-test` or `build-only`: Skip the pytest step
- `--test-only`: Skip build/install, just run tests
- A specific test name: pass it to pytest as `-k <name>`

Run each step sequentially. Report build time, install status, and test summary.
