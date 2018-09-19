# Comm|Scope

CUDA- and NUMA-Aware Multi-CPU / Multi-GPU communication benchmarks for [C3SR Scope](https://github.com/c3sr/scope).

## Contributors

* [Carl Pearson](mailto:pearson@illinois.edu)
* [Sarah Hashash](mailto:hashash2@illinois.edu)

## Documentation

See the `docs` folder for a description of the benchmarks.

# Changelog

## v0.3.0

* Rework documentation
* Use `target_include_scope_directories` and `target_link_scope_libraries`.
* Use Clara for flags.
* Remove numa/rd and numa/wr.

## v0.2.0

* Add `--numa_ids` command line flag.
* Use `--cuda_device_ids` and --`numa_ids` to select CUDA and NUMA devices for benchmarks.

