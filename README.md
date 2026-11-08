# Eurotracer

Eurotracer is a Python project for studying European renewable-power networks: where electricity is generated, how it moves through the grid, and how infrastructure costs can be attributed.

I built it during my MSc in Mechanical Engineering at Aarhus University. My work covered data processing, numerical modelling and comparing results across scenarios. My completed thesis used a 512-node European network. The code here is the rewrite I am continuing to develop.

## Where to start

- [Small example](examples/quickstart.py): a four-node, two-country scenario with synthetic inputs.
- [Model workflow](src/eurotracer/model.py): how the calculation stages fit together.
- [Network calculations](src/eurotracer/network.py): power flow and balance checks.
- [Data processing](src/eurotracer/data.py): input validation and time-series handling.
- [Numerical tests](tests/test_science.py): comparisons against reference calculations and conservation checks.

The main tools are Python, NumPy, SciPy and pandas. This is a compact code sample for browsing; datasets and installation configuration are omitted. The example is illustrative, and validation of the rewrite against the full thesis data is still ongoing.
