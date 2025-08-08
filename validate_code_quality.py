#!/usr/bin/env python3
"""
Code Quality and Structure Validation for DarkOperator Studio Enhancements.

This script validates the code structure, documentation, and implementation quality
without requiring runtime dependencies like PyTorch.
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeQualityValidator:
    """Validates code quality and structure of enhancements."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            'total_files_analyzed': 0,
            'enhancement_components': {},
            'code_quality_metrics': {},
            'documentation_coverage': {},
            'issues_found': [],
            'recommendations': []
        }
        
    def validate_all_enhancements(self) -> Dict[str, Any]:
        """Validate all enhanced components."""
        logger.info("Starting comprehensive code quality validation...")
        
        # Define enhancement components to validate
        enhancement_components = {
            'Security Enhancements': [
                'darkoperator/security/planning_security.py'
            ],
            'Model Hub': [
                'darkoperator/hub/model_hub.py',
                'darkoperator/hub/downloader.py',
                'darkoperator/hub/physics_models.py'
            ],
            'Visualization System': [
                'darkoperator/visualization/interactive.py'
            ],
            'Quantum Scheduler': [
                'darkoperator/planning/quantum_scheduler.py'
            ],
            'Distributed Training': [
                'darkoperator/distributed/gpu_trainer.py'
            ],
            'Benchmarking Suite': [
                'darkoperator/benchmarks/benchmark_runner.py',
                'darkoperator/benchmarks/physics_benchmarks.py'
            ],
            'Global Deployment': [
                'darkoperator/deployment/global_config.py'
            ]
        }
        
        # Validate each component
        for component_name, file_paths in enhancement_components.items():
            logger.info(f"Validating {component_name}...")
            component_results = self.validate_component(component_name, file_paths)
            self.results['enhancement_components'][component_name] = component_results
            
        # Generate overall metrics
        self.generate_overall_metrics()
        
        # Create quality report
        self.generate_quality_report()
        
        return self.results
        
    def validate_component(self, component_name: str, file_paths: List[str]) -> Dict[str, Any]:
        """Validate a specific enhancement component."""
        component_results = {
            'files_found': 0,
            'files_missing': [],
            'total_lines': 0,
            'documentation_score': 0,
            'complexity_score': 0,
            'implementation_quality': 0,
            'issues': [],
            'features_implemented': []
        }
        
        for file_path in file_paths:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                component_results['files_missing'].append(file_path)
                continue
                
            component_results['files_found'] += 1
            self.results['total_files_analyzed'] += 1
            
            # Analyze file
            file_analysis = self.analyze_python_file(full_path)
            component_results['total_lines'] += file_analysis['lines_of_code']
            component_results['issues'].extend(file_analysis['issues'])
            component_results['features_implemented'].extend(file_analysis['features'])
            
            # Update scores
            component_results['documentation_score'] += file_analysis['documentation_score']
            component_results['complexity_score'] += file_analysis['complexity_score'] 
            component_results['implementation_quality'] += file_analysis['implementation_quality']
        
        # Average scores
        if component_results['files_found'] > 0:
            component_results['documentation_score'] /= component_results['files_found']
            component_results['complexity_score'] /= component_results['files_found']
            component_results['implementation_quality'] /= component_results['files_found']
            
        return component_results
        
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for quality metrics."""
        analysis = {
            'lines_of_code': 0,
            'documentation_score': 0,
            'complexity_score': 0,
            'implementation_quality': 0,
            'issues': [],
            'features': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count lines of code (excluding empty lines and comments)
            lines = content.split('\n')
            analysis['lines_of_code'] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                analysis.update(self.analyze_ast(tree, file_path.name))
            except SyntaxError as e:
                analysis['issues'].append(f"Syntax error: {e}")
                
            # Documentation analysis
            analysis['documentation_score'] = self.analyze_documentation(content)
            
            # Feature detection based on file content
            analysis['features'] = self.detect_features(content, file_path.name)
            
        except Exception as e:
            analysis['issues'].append(f"Failed to analyze {file_path}: {e}")
            
        return analysis
        
    def analyze_ast(self, tree: ast.AST, filename: str) -> Dict[str, Any]:
        """Analyze AST for complexity and quality metrics."""
        analysis = {
            'complexity_score': 0,
            'implementation_quality': 0,
            'issues': []
        }
        
        # Count various AST elements
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Quality metrics
        if len(classes) > 0:
            analysis['implementation_quality'] += 20  # Has classes
            
        if len(functions) > 0:
            analysis['implementation_quality'] += 20  # Has functions
            
        # Check for docstrings
        documented_classes = sum(1 for cls in classes if ast.get_docstring(cls))
        documented_functions = sum(1 for func in functions if ast.get_docstring(func))
        
        if documented_classes > 0:
            analysis['implementation_quality'] += 20  # Documented classes
            
        if documented_functions > 0:
            analysis['implementation_quality'] += 20  # Documented functions
            
        # Complexity analysis
        total_nodes = len(list(ast.walk(tree)))
        if total_nodes > 100:
            analysis['complexity_score'] = min(100, total_nodes / 10)  # Cap at 100
        else:
            analysis['complexity_score'] = 50  # Moderate complexity
            
        # Type hints check
        has_type_hints = any(
            func.returns is not None or any(arg.annotation for arg in func.args.args)
            for func in functions
        )
        
        if has_type_hints:
            analysis['implementation_quality'] += 20  # Type hints present
            
        return analysis
        
    def analyze_documentation(self, content: str) -> float:
        """Analyze documentation quality."""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Count docstring lines
        docstring_pattern = r'""".*?"""'
        docstring_matches = re.findall(docstring_pattern, content, re.DOTALL)
        docstring_lines = sum(match.count('\n') + 1 for match in docstring_matches)
        
        # Count comment lines
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Documentation coverage score
        doc_coverage = (docstring_lines + comment_lines) / max(total_lines, 1) * 100
        return min(doc_coverage, 100)  # Cap at 100
        
    def detect_features(self, content: str, filename: str) -> List[str]:
        """Detect implemented features based on content analysis."""
        features = []
        
        # Physics-related features
        if 'conservation' in content.lower():
            features.append('Conservation Laws')
            
        if 'quantum' in content.lower():
            features.append('Quantum Computing')
            
        if 'distributed' in content.lower():
            features.append('Distributed Computing')
            
        if 'gpu' in content.lower():
            features.append('GPU Acceleration')
            
        if 'security' in content.lower() or 'encrypt' in content.lower():
            features.append('Security Enhancements')
            
        if 'benchmark' in content.lower():
            features.append('Benchmarking')
            
        if 'visualization' in content.lower() or 'plot' in content.lower():
            features.append('Visualization')
            
        if 'deployment' in content.lower() or 'terraform' in content.lower():
            features.append('Deployment Automation')
            
        # Advanced features
        if 'async' in content or 'await' in content:
            features.append('Asynchronous Programming')
            
        if 'typing' in content or ': str' in content:
            features.append('Type Annotations')
            
        if 'dataclass' in content:
            features.append('Data Classes')
            
        if 'logging' in content:
            features.append('Logging')
            
        return features
        
    def generate_overall_metrics(self):
        """Generate overall quality metrics."""
        total_files = self.results['total_files_analyzed']
        total_lines = sum(
            comp['total_lines'] 
            for comp in self.results['enhancement_components'].values()
        )
        
        avg_doc_score = sum(
            comp['documentation_score'] 
            for comp in self.results['enhancement_components'].values()
        ) / max(len(self.results['enhancement_components']), 1)
        
        avg_impl_quality = sum(
            comp['implementation_quality'] 
            for comp in self.results['enhancement_components'].values()
        ) / max(len(self.results['enhancement_components']), 1)
        
        # Overall quality assessment
        overall_quality = (avg_doc_score * 0.3 + avg_impl_quality * 0.7)
        
        self.results['code_quality_metrics'] = {
            'total_files_analyzed': total_files,
            'total_lines_of_code': total_lines,
            'average_documentation_score': avg_doc_score,
            'average_implementation_quality': avg_impl_quality,
            'overall_quality_score': overall_quality,
            'quality_grade': self.get_quality_grade(overall_quality)
        }
        
        # Generate recommendations
        self.generate_recommendations(overall_quality)
        
    def get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= 90:
            return 'A+ (Excellent)'
        elif score >= 80:
            return 'A (Very Good)'
        elif score >= 70:
            return 'B (Good)'
        elif score >= 60:
            return 'C (Fair)'
        else:
            return 'D (Needs Improvement)'
            
    def generate_recommendations(self, overall_quality: float):
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if overall_quality >= 85:
            recommendations.extend([
                "✅ Excellent code quality - ready for production deployment",
                "📈 Consider implementing automated code quality gates",
                "🚀 System demonstrates professional-grade architecture"
            ])
        elif overall_quality >= 70:
            recommendations.extend([
                "✅ Good code quality with minor improvements needed",
                "📋 Review any remaining issues and enhance documentation",
                "🔧 Consider refactoring complex functions for better maintainability"
            ])
        else:
            recommendations.extend([
                "⚠️ Code quality needs improvement before production deployment",
                "📚 Increase documentation coverage and add more comments",
                "🔧 Refactor complex code sections and add error handling"
            ])
            
        # Component-specific recommendations
        missing_files = sum(
            len(comp['files_missing']) 
            for comp in self.results['enhancement_components'].values()
        )
        
        if missing_files > 0:
            recommendations.append(f"⚠️ {missing_files} expected enhancement files are missing")
            
        self.results['recommendations'] = recommendations
        
    def generate_quality_report(self):
        """Generate comprehensive quality report."""
        reports_dir = self.project_root / "quality_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # JSON report
        json_report = reports_dir / "code_quality_report.json"
        with open(json_report, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        # Markdown report
        md_report = reports_dir / "code_quality_report.md"
        with open(md_report, 'w') as f:
            f.write("# DarkOperator Studio - Code Quality Assessment Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            metrics = self.results['code_quality_metrics']
            f.write("## Executive Summary\n\n")
            f.write(f"- **Overall Quality Score**: {metrics['overall_quality_score']:.1f}/100\n")
            f.write(f"- **Quality Grade**: {metrics['quality_grade']}\n")
            f.write(f"- **Files Analyzed**: {metrics['total_files_analyzed']}\n")
            f.write(f"- **Total Lines of Code**: {metrics['total_lines_of_code']:,}\n")
            f.write(f"- **Documentation Score**: {metrics['average_documentation_score']:.1f}/100\n")
            f.write(f"- **Implementation Quality**: {metrics['average_implementation_quality']:.1f}/100\n\n")
            
            # Component Analysis
            f.write("## Enhancement Component Analysis\n\n")
            
            for component_name, component_data in self.results['enhancement_components'].items():
                status_icon = "✅" if component_data['files_found'] > 0 else "❌"
                f.write(f"### {status_icon} {component_name}\n\n")
                
                f.write(f"- **Files Found**: {component_data['files_found']}\n")
                if component_data['files_missing']:
                    f.write(f"- **Missing Files**: {', '.join(component_data['files_missing'])}\n")
                f.write(f"- **Lines of Code**: {component_data['total_lines']:,}\n")
                f.write(f"- **Documentation Score**: {component_data['documentation_score']:.1f}/100\n")
                f.write(f"- **Implementation Quality**: {component_data['implementation_quality']:.1f}/100\n")
                
                if component_data['features_implemented']:
                    f.write(f"- **Features Implemented**: {', '.join(set(component_data['features_implemented']))}\n")
                    
                if component_data['issues']:
                    f.write(f"- **Issues Found**: {len(component_data['issues'])}\n")
                    
                f.write("\n")
                
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in self.results['recommendations']:
                f.write(f"- {rec}\n")
                
            f.write("\n")
            
            # Detailed Metrics
            f.write("## Detailed Analysis\n\n")
            f.write("### Code Distribution by Component\n\n")
            
            total_lines = metrics['total_lines_of_code']
            for component_name, component_data in self.results['enhancement_components'].items():
                if total_lines > 0:
                    percentage = (component_data['total_lines'] / total_lines) * 100
                    f.write(f"- **{component_name}**: {component_data['total_lines']:,} lines ({percentage:.1f}%)\n")
                    
        logger.info(f"Quality reports generated:")
        logger.info(f"  - JSON: {json_report}")
        logger.info(f"  - Markdown: {md_report}")


def main():
    """Main validation entry point."""
    print("🔍 Starting DarkOperator Studio Code Quality Validation...")
    print("=" * 60)
    
    validator = CodeQualityValidator()
    results = validator.validate_all_enhancements()
    
    metrics = results['code_quality_metrics']
    
    print("\n" + "=" * 60)
    print("📊 CODE QUALITY SUMMARY")
    print("=" * 60)
    print(f"Overall Quality Score: {metrics['overall_quality_score']:.1f}/100")
    print(f"Quality Grade: {metrics['quality_grade']}")
    print(f"Files Analyzed: {metrics['total_files_analyzed']}")
    print(f"Total Lines of Code: {metrics['total_lines_of_code']:,}")
    print(f"Documentation Score: {metrics['average_documentation_score']:.1f}/100")
    print(f"Implementation Quality: {metrics['average_implementation_quality']:.1f}/100")
    
    if metrics['overall_quality_score'] >= 70:
        print("\n✅ CODE QUALITY PASSED - System meets quality standards!")
    else:
        print("\n⚠️ CODE QUALITY NEEDS IMPROVEMENT - Review recommendations!")
        
    print(f"\n📋 Detailed reports available in: quality_reports/")
    
    return metrics['overall_quality_score'] >= 70


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)