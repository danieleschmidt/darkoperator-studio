"""Main CLI entry point for DarkOperator Studio."""

import argparse
import logging
import sys
from pathlib import Path

import darkoperator as do


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


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DarkOperator Studio: Neural Operators for Dark Matter Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
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