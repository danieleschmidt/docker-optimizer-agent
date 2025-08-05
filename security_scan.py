#!/usr/bin/env python3
"""Security scanning and validation for quantum task planner."""

import os
import sys
import subprocess
import ast
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import tempfile
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Security:
    """Security scanning and validation."""
    
    def __init__(self, project_root: Path):
        """Initialize security scanner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.findings: List[Dict[str, Any]] = []
        self.severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    
    def scan_all(self) -> Dict[str, Any]:
        """Run comprehensive security scan.
        
        Returns:
            Security scan results
        """
        logger.info("Starting comprehensive security scan")
        
        # Run all security checks
        self.scan_hardcoded_secrets()
        self.scan_import_security()
        self.scan_code_injection_risks()
        self.scan_file_permissions()
        self.scan_dependency_vulnerabilities()
        self.scan_quantum_algorithm_security()
        self.validate_error_handling()
        self.check_input_validation()
        
        # Generate report
        return self.generate_report()
    
    def scan_hardcoded_secrets(self) -> None:
        """Scan for hardcoded secrets and credentials."""
        logger.info("Scanning for hardcoded secrets")
        
        # Common secret patterns
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "hardcoded_password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "hardcoded_api_key"),
            (r'secret_key\s*=\s*["\'][^"\']+["\']', "hardcoded_secret_key"),
            (r'token\s*=\s*["\'][^"\']+["\']', "hardcoded_token"),
            (r'(["\'][a-zA-Z0-9]{32,}["\'])', "potential_secret"),
            (r'(sk-[a-zA-Z0-9]{32,})', "openai_api_key"),
            (r'(ghp_[a-zA-Z0-9]{36})', "github_token"),
            (r'(xoxb-[a-zA-Z0-9-]+)', "slack_bot_token")
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            if "test" in str(file_path).lower():
                continue  # Skip test files for now
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip obvious examples and comments
                            if any(x in line.lower() for x in ['example', 'test', 'dummy', 'placeholder', '#']):
                                continue
                                
                            self.add_finding(
                                severity="high",
                                category="secrets",
                                title=f"Potential {secret_type} found",
                                description=f"Line {i} may contain hardcoded secrets",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip()
                            )
            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")
    
    def scan_import_security(self) -> None:
        """Scan for insecure imports and dependencies."""
        logger.info("Scanning import security")
        
        dangerous_imports = {
            'eval': "Use of eval() can execute arbitrary code",
            'exec': "Use of exec() can execute arbitrary code", 
            'subprocess.call': "Direct subprocess calls may be vulnerable to injection",
            'os.system': "os.system() is vulnerable to command injection",
            'pickle': "pickle can execute arbitrary code during deserialization",
            '__import__': "Dynamic imports may be insecure"
        }
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        # Check function calls
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                func_name = node.func.id
                                if func_name in dangerous_imports:
                                    self.add_finding(
                                        severity="medium",
                                        category="code_security",
                                        title=f"Potentially dangerous function: {func_name}",
                                        description=dangerous_imports[func_name],
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=node.lineno
                                    )
                        
                        # Check imports
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                if name.name in dangerous_imports:
                                    self.add_finding(
                                        severity="low",
                                        category="imports",
                                        title=f"Imported potentially dangerous module: {name.name}",
                                        description=dangerous_imports[name.name],
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=node.lineno
                                    )
                
                except SyntaxError:
                    logger.warning(f"Could not parse {file_path} for AST analysis")
                    
            except Exception as e:
                logger.warning(f"Error scanning imports in {file_path}: {e}")
    
    def scan_code_injection_risks(self) -> None:
        """Scan for code injection vulnerabilities."""
        logger.info("Scanning for code injection risks")
        
        injection_patterns = [
            (r'format\s*\([^)]*\%', "string_format_injection"),
            (r'\.format\s*\([^)]*\{[^}]*\}', "format_string_injection"),
            (r'f["\'][^"\']*\{[^}]*\}[^"\']*["\']', "f_string_with_variables"),
            (r'subprocess\.[a-zA-Z]+\s*\([^)]*shell\s*=\s*True', "shell_injection_risk"),
            (r'os\.popen\s*\(', "command_injection_risk")
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, risk_type in injection_patterns:
                        if re.search(pattern, line):
                            self.add_finding(
                                severity="medium",
                                category="injection",
                                title=f"Potential {risk_type}",
                                description="Code may be vulnerable to injection attacks",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip()
                            )
                            
            except Exception as e:
                logger.warning(f"Error scanning {file_path} for injection risks: {e}")
    
    def scan_file_permissions(self) -> None:
        """Scan file permissions for security issues."""
        logger.info("Scanning file permissions")
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check for world-writable files
                    if mode & 0o002:
                        self.add_finding(
                            severity="medium",
                            category="permissions",
                            title="World-writable file",
                            description="File is writable by all users",
                            file_path=str(file_path.relative_to(self.project_root))
                        )
                    
                    # Check for executable scripts without proper shebang
                    if (mode & 0o111) and file_path.suffix == '.py':
                        try:
                            with open(file_path, 'r') as f:
                                first_line = f.readline().strip()
                                if not first_line.startswith('#!'):
                                    self.add_finding(
                                        severity="low",
                                        category="permissions",
                                        title="Executable Python file without shebang",
                                        description="Executable Python file should have proper shebang",
                                        file_path=str(file_path.relative_to(self.project_root))
                                    )
                        except Exception:
                            pass
                            
                except Exception as e:
                    logger.warning(f"Error checking permissions for {file_path}: {e}")
    
    def scan_dependency_vulnerabilities(self) -> None:
        """Scan for known vulnerable dependencies."""
        logger.info("Scanning dependency vulnerabilities")
        
        # Check requirements files
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "pyproject.toml"
        ]
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.check_requirements_file(req_path)
    
    def check_requirements_file(self, req_path: Path) -> None:
        """Check a requirements file for vulnerabilities."""
        try:
            with open(req_path, 'r') as f:
                content = f.read()
            
            # Known vulnerable patterns (simplified for demo)
            vulnerable_packages = {
                'pyyaml<5.4': "PyYAML < 5.4 has arbitrary code execution vulnerability",
                'jinja2<2.11.3': "Jinja2 < 2.11.3 has XSS vulnerabilities",
                'pillow<8.2.0': "Pillow < 8.2.0 has multiple vulnerabilities",
                'urllib3<1.26.5': "urllib3 < 1.26.5 has multiple vulnerabilities"
            }
            
            for vuln_pattern, description in vulnerable_packages.items():
                package_name = vuln_pattern.split('<')[0]
                if package_name in content.lower():
                    # This is a simplified check - real implementation would parse versions
                    self.add_finding(
                        severity="high",
                        category="dependencies",
                        title=f"Potentially vulnerable dependency: {package_name}",
                        description=description,
                        file_path=str(req_path.relative_to(self.project_root))
                    )
                    
        except Exception as e:
            logger.warning(f"Error checking requirements file {req_path}: {e}")
    
    def scan_quantum_algorithm_security(self) -> None:
        """Scan quantum algorithm implementations for security issues."""
        logger.info("Scanning quantum algorithm security")
        
        # Quantum-specific security concerns
        quantum_risks = [
            (r'random\.seed\s*\([^)]*\)', "deterministic_randomness"),
            (r'np\.random\.seed\s*\([^)]*\)', "numpy_deterministic_randomness"),
            (r'quantum_weight\s*=\s*0', "zero_quantum_weight"),
            (r'entanglement_factor\s*=\s*[01]\.0*$', "extreme_entanglement_values"),
            (r'coherence_time\s*=\s*0', "zero_coherence_time")
        ]
        
        algorithm_files = list((self.project_root / "src" / "quantum_task_planner" / "algorithms").glob("*.py"))
        
        for file_path in algorithm_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, risk_type in quantum_risks:
                        if re.search(pattern, line):
                            self.add_finding(
                                severity="low",
                                category="quantum_security",
                                title=f"Quantum algorithm risk: {risk_type}",
                                description="Quantum algorithm may have security implications",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip()
                            )
                            
            except Exception as e:
                logger.warning(f"Error scanning quantum algorithms in {file_path}: {e}")
    
    def validate_error_handling(self) -> None:
        """Validate proper error handling practices."""
        logger.info("Validating error handling")
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # Check for bare except clauses
                for i, line in enumerate(lines, 1):
                    if re.search(r'except\s*:', line.strip()):
                        self.add_finding(
                            severity="medium",
                            category="error_handling",
                            title="Bare except clause",
                            description="Bare except clauses can hide important errors",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            code_snippet=line.strip()
                        )
                    
                    # Check for exception handling without logging
                    if re.search(r'except\s+\w+.*:', line.strip()):
                        # Look ahead for logging in exception block
                        found_logging = False
                        for j in range(i, min(i + 10, len(lines))):
                            if 'log' in lines[j].lower() or 'print' in lines[j].lower():
                                found_logging = True
                                break
                            if re.match(r'^\s*(def|class|if|for|while|try)', lines[j]):
                                break
                        
                        if not found_logging:
                            self.add_finding(
                                severity="low",
                                category="error_handling", 
                                title="Exception without logging",
                                description="Exception caught but not logged",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i
                            )
                            
            except Exception as e:
                logger.warning(f"Error validating error handling in {file_path}: {e}")
    
    def check_input_validation(self) -> None:
        """Check for proper input validation."""
        logger.info("Checking input validation")
        
        # This would be more sophisticated in a real implementation
        python_files = list((self.project_root / "src").glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for functions that accept user input
                function_patterns = [
                    r'def\s+\w+\([^)]*user_input',
                    r'def\s+\w+\([^)]*input_data',
                    r'def\s+\w+\([^)]*request'
                ]
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in function_patterns:
                        if re.search(pattern, line):
                            # This is a simplified check - would need more sophisticated analysis
                            self.add_finding(
                                severity="info",
                                category="input_validation",
                                title="Function accepts user input",  
                                description="Ensure proper input validation is implemented",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                code_snippet=line.strip()
                            )
                            
            except Exception as e:
                logger.warning(f"Error checking input validation in {file_path}: {e}")
    
    def add_finding(self, severity: str, category: str, title: str, 
                   description: str, file_path: str, line_number: int = None,
                   code_snippet: str = None) -> None:
        """Add a security finding.
        
        Args:
            severity: Severity level (critical, high, medium, low, info)
            category: Category of finding
            title: Finding title
            description: Finding description
            file_path: Path to file
            line_number: Line number if applicable
            code_snippet: Code snippet if applicable
        """
        finding = {
            "severity": severity,
            "category": category,
            "title": title,
            "description": description,
            "file_path": file_path,
            "line_number": line_number,
            "code_snippet": code_snippet,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        self.findings.append(finding)
        self.severity_counts[severity] += 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate security scan report.
        
        Returns:
            Security report dictionary
        """
        report = {
            "scan_summary": {
                "total_findings": len(self.findings),
                "severity_breakdown": self.severity_counts.copy(),
                "scan_date": "2024-01-01T00:00:00Z",
                "project_root": str(self.project_root)
            },
            "findings": self.findings,
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings.
        
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        if self.severity_counts["critical"] > 0:
            recommendations.append("🚨 CRITICAL: Address critical security issues immediately")
        
        if self.severity_counts["high"] > 0:
            recommendations.append("⚠️ HIGH: Review and fix high-severity security issues")
        
        if any(f["category"] == "secrets" for f in self.findings):
            recommendations.append("🔐 Use environment variables or secure vaults for secrets")
        
        if any(f["category"] == "injection" for f in self.findings):
            recommendations.append("🛡️ Implement proper input sanitization and validation")
        
        if any(f["category"] == "dependencies" for f in self.findings):
            recommendations.append("📦 Update vulnerable dependencies to secure versions")
        
        if any(f["category"] == "error_handling" for f in self.findings):
            recommendations.append("🐛 Improve error handling and logging practices")
        
        recommendations.extend([
            "🔒 Implement comprehensive input validation",
            "📋 Conduct regular security audits",
            "🧪 Add security-focused unit tests",
            "📚 Security training for development team",
            "🔄 Automated security scanning in CI/CD pipeline"
        ])
        
        return recommendations


def main():
    """Main security scanning function."""
    project_root = Path(__file__).parent
    
    print("🔍 Quantum Task Planner - Security Scanner")
    print("=" * 50)
    
    # Run security scan
    security = Security(project_root)
    report = security.scan_all()
    
    # Display results
    summary = report["scan_summary"]
    print(f"\n📊 Scan Summary:")
    print(f"Total Findings: {summary['total_findings']}")
    print(f"Critical: {summary['severity_breakdown']['critical']}")
    print(f"High: {summary['severity_breakdown']['high']}")
    print(f"Medium: {summary['severity_breakdown']['medium']}")
    print(f"Low: {summary['severity_breakdown']['low']}")
    print(f"Info: {summary['severity_breakdown']['info']}")
    
    # Show findings
    if report["findings"]:
        print(f"\n🔍 Security Findings:")
        for finding in report["findings"][:10]:  # Show first 10
            print(f"\n[{finding['severity'].upper()}] {finding['title']}")
            print(f"  File: {finding['file_path']}")
            if finding['line_number']:
                print(f"  Line: {finding['line_number']}")
            print(f"  Description: {finding['description']}")
            if finding['code_snippet']:
                print(f"  Code: {finding['code_snippet']}")
        
        if len(report["findings"]) > 10:
            print(f"\n... and {len(report['findings']) - 10} more findings")
    
    # Show recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"][:5]:
        print(f"  • {rec}")
    
    # Save report
    report_file = project_root / "security_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Security score
    total_weighted_score = (
        summary['severity_breakdown']['critical'] * 10 +
        summary['severity_breakdown']['high'] * 5 +
        summary['severity_breakdown']['medium'] * 2 +
        summary['severity_breakdown']['low'] * 1
    )
    
    if total_weighted_score == 0:
        security_grade = "A+"
    elif total_weighted_score <= 5:
        security_grade = "A"
    elif total_weighted_score <= 15:
        security_grade = "B"
    elif total_weighted_score <= 30:
        security_grade = "C"
    elif total_weighted_score <= 50:
        security_grade = "D"
    else:
        security_grade = "F"
    
    print(f"\n🎯 Security Grade: {security_grade}")
    print(f"Weighted Score: {total_weighted_score}")
    
    # Return appropriate exit code
    if summary['severity_breakdown']['critical'] > 0:
        return 2
    elif summary['severity_breakdown']['high'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)