# GitHub Actions Workflow Templates

## 🚀 Ready-to-Use Workflow Configurations

The autonomous SDLC enhancement has designed comprehensive GitHub Actions workflows. Due to security restrictions, these need to be manually created in `.github/workflows/`.

## 📋 Workflow Templates

### 1. Core CI/CD Pipeline (ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,security]
    
    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=docker_optimizer --cov-report=xml --cov-fail-under=85
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[security]
    
    - name: Run security checks
      run: |
        bandit -r src/ -f json -o bandit-report.json || true
        safety check --json --output safety-report.json || true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  docker:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build and test Docker images
      run: |
        make docker-build
        make docker-test
```

### 2. Security Scanning (security.yml)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday at 3 AM UTC

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3

  trivy:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Release Automation (release.yml)

See the complete workflow templates in the local `.github/workflows/` directory created by the autonomous enhancement. Each workflow includes:

- **Comprehensive error handling**
- **Security-first design**
- **Performance optimization**
- **Modern GitHub Actions practices**
- **Integration with existing tooling**

## 🔧 Setup Instructions

1. **Create Directory**: `mkdir -p .github/workflows`
2. **Copy Templates**: Use the configurations above or the local files
3. **Configure Secrets**: Add required repository secrets
4. **Test Workflows**: Create a test PR to validate

## 📚 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

---

**Note**: Complete workflow files are available in the autonomous enhancement commit. This document provides templates for manual creation due to GitHub's security restrictions.