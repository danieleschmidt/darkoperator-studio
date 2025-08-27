"""
Simplified Quality Gates for DarkOperator Studio.

This module implements essential quality validation without heavy dependencies.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class QualityGateType(Enum):
    """Types of quality gates."""
    STRUCTURE_VALIDATION = "structure_validation"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION_CHECK = "documentation_check"
    SECURITY_BASICS = "security_basics"
    DEPENDENCY_CHECK = "dependency_check"

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_type: QualityGateType
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

class SimplifiedQualityValidator:
    """Simplified quality validation system."""
    
    def __init__(self):
        self.logger = logging.getLogger("QualityValidator")
        self.results: List[QualityGateResult] = []
        
        # Quality thresholds
        self.thresholds = {
            'structure_completeness': 0.9,
            'code_quality': 0.8,
            'documentation_coverage': 0.7,
            'security_basics': 0.9,
            'dependency_health': 0.8
        }
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        
        print("🚀 Starting Simplified Quality Gates Validation...")
        start_time = time.time()
        
        # Define quality gates
        quality_gates = [
            (self._validate_project_structure, QualityGateType.STRUCTURE_VALIDATION),
            (self._analyze_code_quality, QualityGateType.CODE_ANALYSIS),
            (self._check_documentation, QualityGateType.DOCUMENTATION_CHECK),
            (self._check_security_basics, QualityGateType.SECURITY_BASICS),
            (self._check_dependencies, QualityGateType.DEPENDENCY_CHECK)
        ]
        
        # Execute quality gates
        for gate_func, gate_type in quality_gates:
            try:
                gate_start = time.time()
                result = gate_func()
                result.execution_time = time.time() - gate_start
                result.gate_type = gate_type
                
                self.results.append(result)
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                print(f"  {status} {gate_type.value}: {result.score:.2%}")
                
            except Exception as e:
                print(f"  ❌ FAILED {gate_type.value}: {e}")
                self.results.append(QualityGateResult(
                    gate_type=gate_type,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_results()
        
        print(f"\n🎯 Quality Gates Completed in {total_time:.1f}s")
        print(f"   Overall Score: {analysis['overall_score']:.2%}")
        print(f"   Gates Passed: {analysis['gates_passed']}/{analysis['total_gates']}")
        
        return {
            'execution_time': total_time,
            'results': self.results,
            'analysis': analysis,
            'quality_gates_passed': analysis['overall_passed'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _validate_project_structure(self) -> QualityGateResult:
        """Validate project structure and organization."""
        
        structure_results = {
            'expected_files': 0,
            'found_files': 0,
            'expected_directories': 0,
            'found_directories': 0,
            'structure_details': []
        }
        
        # Expected project structure
        expected_files = [
            'README.md',
            'setup.py',
            'pyproject.toml',
            'requirements.txt',
            'environment.yml',
            'LICENSE'
        ]
        
        expected_directories = [
            'darkoperator',
            'tests',
            'docs',
            'examples'
        ]
        
        # Check files
        for file_path in expected_files:
            structure_results['expected_files'] += 1
            full_path = f'/root/repo/{file_path}'
            
            if os.path.exists(full_path):
                structure_results['found_files'] += 1
                file_size = os.path.getsize(full_path)
                status = 'found'
            else:
                file_size = 0
                status = 'missing'
            
            structure_results['structure_details'].append({
                'type': 'file',
                'path': file_path,
                'status': status,
                'size_bytes': file_size
            })
        
        # Check directories
        for dir_path in expected_directories:
            structure_results['expected_directories'] += 1
            full_path = f'/root/repo/{dir_path}'
            
            if os.path.exists(full_path) and os.path.isdir(full_path):
                structure_results['found_directories'] += 1
                status = 'found'
                
                # Count files in directory
                try:
                    file_count = len([f for f in os.listdir(full_path) 
                                    if os.path.isfile(os.path.join(full_path, f))])
                except:
                    file_count = 0
            else:
                status = 'missing'
                file_count = 0
            
            structure_results['structure_details'].append({
                'type': 'directory',
                'path': dir_path,
                'status': status,
                'file_count': file_count
            })
        
        # Calculate completeness score
        file_score = structure_results['found_files'] / max(structure_results['expected_files'], 1)
        dir_score = structure_results['found_directories'] / max(structure_results['expected_directories'], 1)
        overall_score = (file_score + dir_score) / 2
        
        return QualityGateResult(
            gate_type=QualityGateType.STRUCTURE_VALIDATION,
            passed=overall_score >= self.thresholds['structure_completeness'],
            score=overall_score,
            details=structure_results,
            execution_time=0.0
        )
    
    def _analyze_code_quality(self) -> QualityGateResult:
        """Analyze code quality metrics."""
        
        quality_results = {
            'python_files_analyzed': 0,
            'total_lines': 0,
            'docstring_coverage': 0.0,
            'avg_function_length': 0.0,
            'quality_metrics': []
        }
        
        # Analyze Python files in darkoperator directory
        darkoperator_path = '/root/repo/darkoperator'
        python_files = []
        
        if os.path.exists(darkoperator_path):
            for root, dirs, files in os.walk(darkoperator_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
        
        total_functions = 0
        functions_with_docstrings = 0
        total_function_lines = 0
        
        for file_path in python_files[:10]:  # Analyze first 10 files for demo
            try:
                quality_results['python_files_analyzed'] += 1
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    quality_results['total_lines'] += len(lines)
                
                # Simple analysis for functions and docstrings
                in_function = False
                current_function_lines = 0
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Detect function definition
                    if stripped.startswith('def ') or stripped.startswith('async def '):
                        if in_function:
                            total_function_lines += current_function_lines
                        
                        in_function = True
                        current_function_lines = 1
                        total_functions += 1
                        
                        # Check for docstring in next few lines
                        has_docstring = False
                        for j in range(i+1, min(i+5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line.startswith('"""') or next_line.startswith("'''"):
                                has_docstring = True
                                break
                            elif next_line and not next_line.startswith('#'):
                                break
                        
                        if has_docstring:
                            functions_with_docstrings += 1
                    
                    elif in_function:
                        current_function_lines += 1
                        
                        # End of function detection (simplified)
                        if stripped and not line.startswith('    ') and not line.startswith('\t'):
                            if not (stripped.startswith('def ') or stripped.startswith('async def ')):
                                total_function_lines += current_function_lines
                                in_function = False
                                current_function_lines = 0
                
                # Handle last function
                if in_function:
                    total_function_lines += current_function_lines
                
            except Exception as e:
                quality_results['quality_metrics'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        # Calculate metrics
        if total_functions > 0:
            quality_results['docstring_coverage'] = functions_with_docstrings / total_functions
            quality_results['avg_function_length'] = total_function_lines / total_functions
        
        # Calculate overall quality score
        docstring_score = quality_results['docstring_coverage']
        
        # Penalize very long functions
        length_penalty = max(0, min(1, 20 / max(quality_results['avg_function_length'], 5)))
        
        # File count bonus
        file_bonus = min(1.0, quality_results['python_files_analyzed'] / 10)
        
        overall_score = (docstring_score * 0.5 + length_penalty * 0.3 + file_bonus * 0.2)
        
        return QualityGateResult(
            gate_type=QualityGateType.CODE_ANALYSIS,
            passed=overall_score >= self.thresholds['code_quality'],
            score=overall_score,
            details=quality_results,
            execution_time=0.0
        )
    
    def _check_documentation(self) -> QualityGateResult:
        """Check documentation completeness."""
        
        doc_results = {
            'readme_score': 0.0,
            'docstring_presence': 0.0,
            'examples_present': False,
            'docs_directory_present': False,
            'documentation_details': []
        }
        
        # Check README.md
        readme_path = '/root/repo/README.md'
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Score README based on sections
                expected_sections = [
                    'installation', 'usage', 'example', 'quick start',
                    'overview', 'features', 'citation', 'license'
                ]
                
                sections_found = 0
                for section in expected_sections:
                    if section.lower() in readme_content.lower():
                        sections_found += 1
                
                doc_results['readme_score'] = sections_found / len(expected_sections)
                
                doc_results['documentation_details'].append({
                    'type': 'README',
                    'sections_found': sections_found,
                    'total_sections': len(expected_sections),
                    'word_count': len(readme_content.split())
                })
                
            except Exception as e:
                doc_results['documentation_details'].append({
                    'type': 'README',
                    'error': str(e)
                })
        
        # Check examples directory
        examples_path = '/root/repo/examples'
        if os.path.exists(examples_path):
            doc_results['examples_present'] = True
            try:
                example_files = [f for f in os.listdir(examples_path) if f.endswith('.py')]
                doc_results['documentation_details'].append({
                    'type': 'examples',
                    'file_count': len(example_files)
                })
            except:
                pass
        
        # Check docs directory
        docs_path = '/root/repo/docs'
        if os.path.exists(docs_path):
            doc_results['docs_directory_present'] = True
        
        # Calculate overall documentation score
        readme_weight = 0.5
        examples_weight = 0.3
        docs_weight = 0.2
        
        overall_score = (
            doc_results['readme_score'] * readme_weight +
            (1.0 if doc_results['examples_present'] else 0.0) * examples_weight +
            (1.0 if doc_results['docs_directory_present'] else 0.0) * docs_weight
        )
        
        return QualityGateResult(
            gate_type=QualityGateType.DOCUMENTATION_CHECK,
            passed=overall_score >= self.thresholds['documentation_coverage'],
            score=overall_score,
            details=doc_results,
            execution_time=0.0
        )
    
    def _check_security_basics(self) -> QualityGateResult:
        """Check basic security practices."""
        
        security_results = {
            'checks_performed': 0,
            'checks_passed': 0,
            'security_details': []
        }
        
        # Check for .gitignore file
        gitignore_path = '/root/repo/.gitignore'
        security_results['checks_performed'] += 1
        
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r') as f:
                    gitignore_content = f.read()
                
                # Check for sensitive patterns in .gitignore
                sensitive_patterns = ['*.key', '*.pem', '.env', '__pycache__', '*.pyc']
                patterns_found = sum(1 for pattern in sensitive_patterns if pattern in gitignore_content)
                
                gitignore_good = patterns_found >= 3
                if gitignore_good:
                    security_results['checks_passed'] += 1
                
                security_results['security_details'].append({
                    'check': 'gitignore_security',
                    'passed': gitignore_good,
                    'patterns_found': patterns_found,
                    'total_patterns': len(sensitive_patterns)
                })
            except Exception as e:
                security_results['security_details'].append({
                    'check': 'gitignore_security',
                    'passed': False,
                    'error': str(e)
                })
        else:
            security_results['security_details'].append({
                'check': 'gitignore_security',
                'passed': False,
                'reason': 'gitignore_missing'
            })
        
        # Check for obvious hardcoded secrets (simplified)
        security_results['checks_performed'] += 1
        
        secret_scan_results = self._scan_for_secrets()
        secrets_clean = secret_scan_results['secrets_found'] == 0
        
        if secrets_clean:
            security_results['checks_passed'] += 1
        
        security_results['security_details'].append({
            'check': 'hardcoded_secrets',
            'passed': secrets_clean,
            'secrets_found': secret_scan_results['secrets_found'],
            'files_scanned': secret_scan_results['files_scanned']
        })
        
        # Check requirements.txt for known vulnerable packages
        security_results['checks_performed'] += 1
        
        requirements_secure = self._check_requirements_security()
        if requirements_secure:
            security_results['checks_passed'] += 1
        
        security_results['security_details'].append({
            'check': 'requirements_security',
            'passed': requirements_secure,
            'details': 'basic_vulnerability_check'
        })
        
        # Calculate security score
        security_score = security_results['checks_passed'] / max(security_results['checks_performed'], 1)
        
        return QualityGateResult(
            gate_type=QualityGateType.SECURITY_BASICS,
            passed=security_score >= self.thresholds['security_basics'],
            score=security_score,
            details=security_results,
            execution_time=0.0
        )
    
    def _scan_for_secrets(self) -> Dict[str, Any]:
        """Scan for potential hardcoded secrets."""
        
        secrets_found = 0
        files_scanned = 0
        
        # Patterns that might indicate secrets
        secret_patterns = [
            'password = ',
            'api_key = ',
            'secret = ',
            'token = ',
            'private_key = '
        ]
        
        # Scan Python files
        for root, dirs, files in os.walk('/root/repo'):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            
                        files_scanned += 1
                        
                        for pattern in secret_patterns:
                            if pattern in content:
                                # Additional check to avoid false positives
                                lines = content.split('\n')
                                for line in lines:
                                    if pattern in line and not line.strip().startswith('#'):
                                        # Check if it looks like a real secret (not a placeholder)
                                        if 'example' not in line and 'placeholder' not in line:
                                            secrets_found += 1
                                            break
                                break
                    except:
                        pass
                
                if files_scanned >= 50:  # Limit scan for performance
                    break
            
            if files_scanned >= 50:
                break
        
        return {
            'secrets_found': secrets_found,
            'files_scanned': files_scanned
        }
    
    def _check_requirements_security(self) -> bool:
        """Check requirements.txt for basic security."""
        
        requirements_path = '/root/repo/requirements.txt'
        
        if not os.path.exists(requirements_path):
            return True  # No requirements file is okay
        
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.read().lower()
            
            # Check for obviously vulnerable or outdated patterns
            suspicious_packages = [
                'django==1.',  # Very old Django versions
                'flask==0.',   # Very old Flask versions
                'requests==2.0',  # Very old requests
            ]
            
            for package in suspicious_packages:
                if package in requirements:
                    return False
            
            return True
            
        except:
            return True
    
    def _check_dependencies(self) -> QualityGateResult:
        """Check dependency health and management."""
        
        dep_results = {
            'management_files': 0,
            'expected_files': 0,
            'dependency_details': []
        }
        
        # Expected dependency management files
        expected_files = [
            ('requirements.txt', 'pip_requirements'),
            ('environment.yml', 'conda_environment'),
            ('pyproject.toml', 'modern_python_project'),
            ('setup.py', 'traditional_setup')
        ]
        
        for filename, file_type in expected_files:
            dep_results['expected_files'] += 1
            file_path = f'/root/repo/{filename}'
            
            if os.path.exists(file_path):
                dep_results['management_files'] += 1
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Count dependencies
                    if filename == 'requirements.txt':
                        dep_count = len([line for line in content.split('\n') 
                                       if line.strip() and not line.strip().startswith('#')])
                    elif filename == 'environment.yml':
                        dep_count = content.count('- ')
                    else:
                        dep_count = content.count('==') + content.count('>=')
                    
                    dep_results['dependency_details'].append({
                        'file': filename,
                        'type': file_type,
                        'present': True,
                        'dependency_count': dep_count,
                        'file_size': len(content)
                    })
                    
                except Exception as e:
                    dep_results['dependency_details'].append({
                        'file': filename,
                        'type': file_type,
                        'present': True,
                        'error': str(e)
                    })
            else:
                dep_results['dependency_details'].append({
                    'file': filename,
                    'type': file_type,
                    'present': False
                })
        
        # Calculate dependency health score
        management_score = dep_results['management_files'] / dep_results['expected_files']
        
        # Bonus for having multiple dependency management approaches
        variety_bonus = min(0.2, dep_results['management_files'] * 0.05)
        
        overall_score = min(1.0, management_score + variety_bonus)
        
        return QualityGateResult(
            gate_type=QualityGateType.DEPENDENCY_CHECK,
            passed=overall_score >= self.thresholds['dependency_health'],
            score=overall_score,
            details=dep_results,
            execution_time=0.0
        )
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze quality gate results."""
        
        if not self.results:
            return {
                'overall_passed': False,
                'overall_score': 0.0,
                'gates_passed': 0,
                'total_gates': 0,
                'critical_failures': [],
                'recommendations': ['No quality gates executed']
            }
        
        total_gates = len(self.results)
        gates_passed = sum(1 for result in self.results if result.passed)
        overall_score = sum(result.score for result in self.results) / total_gates
        
        # Identify critical failures
        critical_failures = []
        for result in self.results:
            if not result.passed and result.score < 0.5:  # Very low score
                critical_failures.append({
                    'gate': result.gate_type.value,
                    'score': result.score,
                    'error': result.error_message
                })
        
        # Generate recommendations
        recommendations = []
        for result in self.results:
            if not result.passed:
                if result.gate_type == QualityGateType.STRUCTURE_VALIDATION:
                    recommendations.append("Complete missing project files and directories")
                elif result.gate_type == QualityGateType.CODE_ANALYSIS:
                    recommendations.append("Improve code quality and documentation")
                elif result.gate_type == QualityGateType.DOCUMENTATION_CHECK:
                    recommendations.append("Enhance project documentation")
                elif result.gate_type == QualityGateType.SECURITY_BASICS:
                    recommendations.append("Address basic security issues")
                elif result.gate_type == QualityGateType.DEPENDENCY_CHECK:
                    recommendations.append("Improve dependency management")
        
        # Overall pass/fail decision
        overall_passed = (
            gates_passed / total_gates >= 0.8 and  # 80% of gates must pass
            len(critical_failures) == 0 and        # No critical failures
            overall_score >= 0.7                   # Overall score threshold
        )
        
        return {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'gates_passed': gates_passed,
            'total_gates': total_gates,
            'pass_rate': gates_passed / total_gates,
            'critical_failures': critical_failures,
            'recommendations': recommendations,
            'gate_scores': {
                result.gate_type.value: result.score for result in self.results
            }
        }

def main():
    """Main function to run simplified quality gates."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run quality validation
    validator = SimplifiedQualityValidator()
    
    try:
        results = validator.run_all_quality_gates()
        
        # Save results to file
        with open('/root/repo/quality_gates_final_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print final summary
        analysis = results['analysis']
        
        print("\n" + "="*60)
        print("🎯 QUALITY GATES FINAL REPORT")
        print("="*60)
        print(f"Overall Status: {'✅ PASSED' if analysis['overall_passed'] else '❌ FAILED'}")
        print(f"Overall Score: {analysis['overall_score']:.2%}")
        print(f"Gates Passed: {analysis['gates_passed']}/{analysis['total_gates']}")
        print(f"Execution Time: {results['execution_time']:.1f}s")
        
        print(f"\n📊 Quality Gate Scores:")
        for gate, score in analysis['gate_scores'].items():
            status = "✅" if score >= 0.7 else "❌"
            print(f"  {status} {gate}: {score:.2%}")
        
        if analysis['critical_failures']:
            print(f"\n⚠️ Critical Failures:")
            for failure in analysis['critical_failures']:
                print(f"  - {failure['gate']}: {failure.get('error', 'Failed')}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n✨ Quality gates validation completed!")
        print(f"Report saved to: quality_gates_final_report.json")
        
        return analysis['overall_passed']
        
    except Exception as e:
        print(f"❌ Quality gates validation failed: {e}")
        logging.exception("Quality gates validation error")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)