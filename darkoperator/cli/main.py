"""
Main CLI entry point for DarkOperator Studio.

Enhanced with TERRAGON SDLC v4.0 Autonomous Execution capabilities.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import darkoperator as do

# Import autonomous execution capabilities
try:
    from darkoperator.core import get_autonomous_executor, run_autonomous_sdlc
    HAS_AUTONOMOUS = True
except ImportError:
    HAS_AUTONOMOUS = False


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_list_datasets(args) -> None:
    """List available LHC Open Data datasets."""
    datasets = do.list_opendata_datasets(
        experiment=args.experiment,
        years=args.years,
        data_type=args.data_type
    )
    
    print(f"\nFound {len(datasets)} datasets:")
    print("-" * 80)
    for ds in datasets:
        print(f"Name: {ds['name']}")
        print(f"Experiment: {ds['experiment']} ({ds['year']})")
        print(f"Type: {ds['data_type']}, Energy: {ds['energy']}")
        print(f"Events: {ds['n_events']:,}, Size: {ds['size_gb']:.1f} GB")
        print(f"URL: {ds['url']}")
        print("-" * 80)


def cmd_download(args) -> None:
    """Download dataset."""
    path = do.download_dataset(
        args.dataset,
        cache_dir=args.output_dir,
        max_events=args.max_events,
        force_download=args.force
    )
    print(f"Dataset downloaded to: {path}")


def cmd_analyze(args) -> None:
    """Run anomaly detection analysis."""
    print(f"Loading dataset: {args.dataset}")
    events = do.load_opendata(args.dataset, max_events=args.max_events)
    
    print(f"Loaded {len(events)} events")
    print("Setting up calorimeter operator...")
    
    # Load or create operator
    if args.operator_checkpoint:
        calo_op = do.CalorimeterOperator.from_pretrained(args.operator_checkpoint)
    else:
        calo_op = do.CalorimeterOperator()
        print("Warning: Using untrained operator. Results will not be meaningful.")
    
    # Setup anomaly detector
    print("Initializing conformal detector...")
    detector = do.ConformalDetector(
        operator=calo_op,
        alpha=args.alpha
    )
    
    # Split data for calibration and testing
    n_cal = int(len(events) * 0.5)
    cal_events = events[:n_cal]
    test_events = events[n_cal:]
    
    print(f"Calibrating on {len(cal_events)} events...")
    detector.calibrate(cal_events)
    
    print(f"Analyzing {len(test_events)} test events...")
    anomalies, p_values = detector.find_anomalies(test_events, return_scores=True)
    
    print(f"\nFound {len(anomalies)} anomalies (α = {args.alpha:.2e})")
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "anomaly_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Anomaly Detection Results\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"False Discovery Rate: {args.alpha:.2e}\n")
            f.write(f"Anomalies found: {len(anomalies)}/{len(test_events)}\n\n")
            
            for i, (idx, pval) in enumerate(zip(anomalies[:10], p_values[:10])):
                f.write(f"Anomaly {i+1}: Event {idx}, p-value = {pval:.2e}\n")
        
        print(f"Results saved to: {results_file}")


def cmd_autonomous(args) -> None:
    """Launch autonomous TERRAGON SDLC execution."""
    if not HAS_AUTONOMOUS:
        print("❌ Autonomous execution capabilities not available")
        print("Install with: pip install darkoperator[autonomous]")
        sys.exit(1)
        
    print("🚀 TERRAGON SDLC v4.0 - Autonomous Execution Mode")
    print("=" * 60)
    
    if args.report_only:
        # Generate and display current report
        executor = get_autonomous_executor()
        report = asyncio.run(executor.generate_autonomous_report())
        
        print(json.dumps(report, indent=2))
        
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(f"terragon_report_{timestamp}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📊 Report saved to: {report_file}")
            
    else:
        # Start autonomous execution cycle
        print("Starting autonomous enhancement cycle...")
        print("Press Ctrl+C to stop")
        
        try:
            asyncio.run(run_autonomous_sdlc())
        except KeyboardInterrupt:
            print("\n⏹️  Autonomous execution stopped by user")
        except Exception as e:
            print(f"\n❌ Autonomous execution failed: {e}")
            sys.exit(1)


def cmd_quality_gates(args) -> None:
    """Run quality gates assessment."""
    if not HAS_AUTONOMOUS:
        print("❌ Quality gates require autonomous execution capabilities")
        sys.exit(1)
        
    print("🛡️ TERRAGON SDLC v4.0 - Quality Gates Assessment")
    print("=" * 60)
    
    executor = get_autonomous_executor()
    
    gates_to_run = args.gates if args.gates else [
        'performance', 'security', 'physics_accuracy', 'scalability', 'global_compliance'
    ]
    
    results = []
    
    for gate_name in gates_to_run:
        print(f"\n⏳ Running {gate_name} gate...")
        start_time = time.time()
        
        result = asyncio.run(executor.execute_quality_gate(gate_name))
        result.execution_time = time.time() - start_time
        results.append(result)
        
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"{status} - Score: {result.score:.3f} ({result.execution_time:.2f}s)")
        
        if result.recommendations:
            print("💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    
    passed_gates = sum(1 for r in results if r.passed)
    total_gates = len(results)
    success_rate = passed_gates / total_gates if total_gates > 0 else 0
    avg_score = sum(r.score for r in results) / total_gates if total_gates > 0 else 0
    
    print(f"Gates Passed: {passed_gates}/{total_gates} ({success_rate:.1%})")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Overall Status: {'✅ PRODUCTION READY' if success_rate >= 0.85 else '⚠️  NEEDS IMPROVEMENT'}")
    
    # Save detailed results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"quality_gates_{timestamp}.json")
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed_gates': passed_gates,
                'total_gates': total_gates,
                'success_rate': success_rate,
                'average_score': avg_score
            },
            'detailed_results': [
                {
                    'gate_name': r.gate_name,
                    'passed': r.passed,
                    'score': r.score,
                    'execution_time': r.execution_time,
                    'details': r.details,
                    'recommendations': r.recommendations
                }
                for r in results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"📄 Detailed results saved to: {results_file}")


def main() -> None:
    """Main CLI entry point with enhanced TERRAGON SDLC v4.0 capabilities."""
    parser = argparse.ArgumentParser(
        description="DarkOperator Studio: Neural Operators for Dark Matter Detection\n"
                   "Enhanced with TERRAGON SDLC v4.0 Autonomous Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  darkoperator list                              # List available datasets
  darkoperator download cms-jets-2015            # Download dataset
  darkoperator analyze cms-jets-2015             # Run anomaly detection
  darkoperator autonomous --report-only          # Generate autonomous report
  darkoperator autonomous                        # Start autonomous execution
  darkoperator quality-gates                     # Run all quality gates
  darkoperator quality-gates --gates performance security  # Run specific gates
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List datasets command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("--experiment", nargs="+", help="Filter by experiment")
    list_parser.add_argument("--years", type=int, nargs="+", help="Filter by years")
    list_parser.add_argument("--data-type", help="Filter by data type")
    list_parser.set_defaults(func=cmd_list_datasets)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset")
    download_parser.add_argument("dataset", help="Dataset name")
    download_parser.add_argument("--output-dir", default="./data", help="Output directory")
    download_parser.add_argument("--max-events", type=int, help="Maximum events to download")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    download_parser.set_defaults(func=cmd_download)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run anomaly detection")
    analyze_parser.add_argument("dataset", help="Dataset name")
    analyze_parser.add_argument("--max-events", type=int, default=10000, help="Maximum events")
    analyze_parser.add_argument("--alpha", type=float, default=1e-6, help="False discovery rate")
    analyze_parser.add_argument("--operator-checkpoint", help="Pre-trained operator checkpoint")
    analyze_parser.add_argument("--output-dir", help="Output directory for results")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Autonomous execution command  
    autonomous_parser = subparsers.add_parser("autonomous", help="TERRAGON SDLC v4.0 Autonomous Execution")
    autonomous_parser.add_argument("--report-only", action="store_true", help="Generate report without starting execution")
    autonomous_parser.add_argument("--save-report", action="store_true", help="Save report to file")
    autonomous_parser.set_defaults(func=cmd_autonomous)
    
    # Quality gates command
    quality_parser = subparsers.add_parser("quality-gates", help="Run quality gates assessment")
    quality_parser.add_argument(
        "--gates", 
        nargs="+", 
        choices=["performance", "security", "physics_accuracy", "scalability", "global_compliance"],
        help="Specific gates to run (default: all)"
    )
    quality_parser.add_argument("--save-results", action="store_true", help="Save detailed results to file")
    quality_parser.set_defaults(func=cmd_quality_gates)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logging.error(f"Command failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()