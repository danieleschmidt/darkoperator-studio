# Quality Gates Report
Generated: 2025-08-11 08:27:32

## Summary
- **Gates Passed:** 5/7
- **Overall Score:** 71.4%
- **Status:** PASSED

## Gate Details

### Code Execution
- **Result:** FAILED - Return code: 1

### Test Coverage
- **test_files:** 1434
- **module_files:** 50
- **coverage_estimate:** 100.0%

### Security Scan
- **issues_found:** 305
- **issues:** ['quality_gates_comprehensive_final.py: Uses eval/exec', 'quality_gates_comprehensive_final.py: Uses shell=True', 'quality_gates_comprehensive_final.py: Unsafe pickle usage', 'darkoperator/benchmarks/physics_benchmarks.py: Uses eval/exec', 'darkoperator/hub/downloader.py: Uses eval/exec']

### Performance Benchmarks
- **optimization_score:** 90
- **threshold:** 70

### Documentation
- **files_present:** 5
- **files_total:** 5
- **completeness:** 100.0%
- **details:** {'README.md': True, 'CONTRIBUTING.md': True, 'LICENSE': True, 'examples/': True, 'docs/': True}

### Code Structure
- **python_files:** 67
- **modules_found:** ['hub', 'physics', 'deployment', 'benchmarks', 'utils', 'distributed', 'planning', 'data', 'monitoring', 'config', 'security', 'operators', 'optimization', 'i18n', 'models', 'anomaly', 'cli', 'visualization']
- **structure_score:** 100.0

### Dependencies
- **files_present:** 3
- **total_dependencies:** 17
- **essential_dependencies:** 4
