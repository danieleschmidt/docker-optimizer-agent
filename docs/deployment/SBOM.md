# Software Bill of Materials (SBOM) Generation

## Overview

The Docker Optimizer Agent generates comprehensive Software Bill of Materials (SBOM) documents to support supply chain security, compliance requirements, and vulnerability management. SBOMs provide detailed inventory of all software components, dependencies, and their metadata.

## Supported SBOM Formats

### SPDX (Software Package Data Exchange)
- **Format**: JSON, YAML, Tag-Value, RDF
- **Standard**: ISO/IEC 5962:2021
- **Use Case**: Industry standard compliance
- **Tools**: `spdx-tools`, `cyclonedx-python`

### CycloneDX
- **Format**: JSON, XML, Protocol Buffers
- **Standard**: OWASP CycloneDX
- **Use Case**: Security-focused analysis
- **Tools**: `cyclonedx-bom`, `cyclonedx-python`

### SWID (Software Identification)
- **Format**: XML
- **Standard**: ISO/IEC 19770-2
- **Use Case**: Software asset management
- **Tools**: `swid-tools`

## SBOM Generation Methods

### 1. Python Package SBOM

Generate SBOM for Python dependencies:

```bash
# Generate SPDX JSON format
pip install cyclonedx-bom
cyclonedx-py --format json --output sbom/python-dependencies.spdx.json

# Generate CycloneDX format
cyclonedx-py --format xml --output sbom/python-dependencies.cyclonedx.xml

# Include development dependencies
cyclonedx-py --format json --include-dev --output sbom/python-full.cyclonedx.json
```

### 2. Container Image SBOM

Generate SBOM for Docker images:

```bash
# Using Syft (Anchore)
syft docker-optimizer:latest -o spdx-json > sbom/container-image.spdx.json
syft docker-optimizer:latest -o cyclonedx-json > sbom/container-image.cyclonedx.json

# Using Docker Scout (requires Docker Desktop)
docker scout sbom docker-optimizer:latest --format spdx > sbom/scout-image.spdx.json

# Using Trivy
trivy image --format spdx-json docker-optimizer:latest > sbom/trivy-image.spdx.json
```

### 3. Source Code SBOM

Generate SBOM from source code analysis:

```bash
# Using cdxgen (comprehensive)
npx @cyclonedx/cdxgen -t python -o sbom/source-code.cyclonedx.json .

# Using tern for Dockerfile analysis
tern report -f spdxjson -i docker-optimizer:latest -o sbom/dockerfile.spdx.json
```

## Automated SBOM Generation

### Build Integration

Add SBOM generation to the build process:

```bash
# In scripts/build.sh
generate_sbom() {
    local image_name="$1"
    local output_dir="sbom"
    
    mkdir -p "$output_dir"
    
    echo "Generating SBOM for $image_name..."
    
    # Python dependencies SBOM
    cyclonedx-py --format json --output "$output_dir/python-deps.cyclonedx.json"
    
    # Container image SBOM
    syft "$image_name" -o spdx-json > "$output_dir/container.spdx.json"
    syft "$image_name" -o cyclonedx-json > "$output_dir/container.cyclonedx.json"
    
    # Source code SBOM
    npx @cyclonedx/cdxgen -t python -o "$output_dir/source.cyclonedx.json" .
    
    echo "SBOM files generated in $output_dir/"
}

# Usage
generate_sbom "docker-optimizer:latest"
```

### Makefile Integration

```makefile
# Add to Makefile
.PHONY: sbom sbom-clean sbom-validate

sbom: ## Generate all SBOM documents
	@echo "Generating Software Bill of Materials..."
	@mkdir -p sbom
	@cyclonedx-py --format json --output sbom/python-deps.cyclonedx.json
	@syft docker-optimizer:latest -o spdx-json > sbom/container.spdx.json
	@npx @cyclonedx/cdxgen -t python -o sbom/source.cyclonedx.json .
	@echo "SBOM generation complete. Files in sbom/"

sbom-clean: ## Clean SBOM files
	@rm -rf sbom/

sbom-validate: ## Validate SBOM files
	@echo "Validating SBOM files..."
	@spdx-tools -f json -v sbom/container.spdx.json
	@cyclonedx-py validate --input-file sbom/python-deps.cyclonedx.json
```

## SBOM Validation and Quality

### Validation Tools

```bash
# Validate SPDX documents
pip install spdx-tools
spdx-tools -f json -v sbom/container.spdx.json

# Validate CycloneDX documents
cyclonedx-py validate --input-file sbom/python-deps.cyclonedx.json

# Validate with external services
curl -X POST https://sbom.example.com/validate \
  -H "Content-Type: application/json" \
  -d @sbom/container.cyclonedx.json
```

### Quality Metrics

Track SBOM quality metrics:

```bash
#!/bin/bash
# sbom-quality-check.sh

analyze_sbom_quality() {
    local sbom_file="$1"
    
    echo "SBOM Quality Analysis: $sbom_file"
    echo "========================================="
    
    # Component count
    components=$(jq '.components | length' "$sbom_file")
    echo "Total Components: $components"
    
    # License coverage
    licensed=$(jq '[.components[] | select(.licenses != null)] | length' "$sbom_file")
    license_coverage=$((licensed * 100 / components))
    echo "License Coverage: $license_coverage%"
    
    # Vulnerability data
    vuln_data=$(jq '[.components[] | select(.vulnerabilities != null)] | length' "$sbom_file")
    echo "Components with Vulnerability Data: $vuln_data"
    
    # Hash verification
    hashed=$(jq '[.components[] | select(.hashes != null)] | length' "$sbom_file")
    hash_coverage=$((hashed * 100 / components))
    echo "Hash Coverage: $hash_coverage%"
}

analyze_sbom_quality "sbom/container.cyclonedx.json"
```

## SBOM Integration with Security Tools

### Vulnerability Scanning

```bash
# Use SBOM for targeted vulnerability scanning
grype sbom:sbom/container.spdx.json -o json > vulnerability-report.json

# Cross-reference with NIST NVD
osv-scanner --sbom sbom/container.cyclonedx.json

# Supply chain risk analysis
bomber scan sbom/container.cyclonedx.json
```

### License Compliance

```bash
# License analysis from SBOM
fossology-cli analyze --sbom sbom/container.spdx.json

# SPDX license compliance
spdx-tools -f json --check sbom/container.spdx.json

# Custom license policy checking
jq '.components[] | select(.licenses[] | .license.id | test("GPL|AGPL"))' \
   sbom/container.cyclonedx.json > gpl-components.json
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/sbom.yml (documentation template)
name: SBOM Generation
on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install SBOM tools
        run: |
          pip install cyclonedx-bom
          curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
          
      - name: Build Docker image
        run: docker build -t docker-optimizer:latest .
        
      - name: Generate SBOM
        run: |
          mkdir -p sbom
          cyclonedx-py --format json --output sbom/python-deps.cyclonedx.json
          syft docker-optimizer:latest -o spdx-json > sbom/container.spdx.json
          syft docker-optimizer:latest -o cyclonedx-json > sbom/container.cyclonedx.json
          
      - name: Validate SBOM
        run: |
          cyclonedx-py validate --input-file sbom/python-deps.cyclonedx.json
          # Add more validation as needed
          
      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: sbom-documents
          path: sbom/
          
      - name: Archive SBOM in release
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v1
        with:
          files: sbom/*
```

### GitLab CI

```yaml
# .gitlab-ci.yml template section
generate-sbom:
  stage: build
  image: python:3.11
  before_script:
    - pip install cyclonedx-bom
    - curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
  script:
    - mkdir -p sbom
    - cyclonedx-py --format json --output sbom/python-deps.cyclonedx.json
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - syft $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -o spdx-json > sbom/container.spdx.json
  artifacts:
    paths:
      - sbom/
    expire_in: 30 days
```

## SBOM Storage and Distribution

### Artifact Registry

```bash
# Push SBOM to OCI registry
oras push registry.example.com/docker-optimizer/sbom:latest \
  sbom/container.spdx.json:application/spdx+json \
  sbom/container.cyclonedx.json:application/vnd.cyclonedx+json

# Pull SBOM from registry
oras pull registry.example.com/docker-optimizer/sbom:latest
```

### SBOM Database

```bash
# Store SBOM in database for querying
curl -X POST https://sbom-db.example.com/api/v1/sbom \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d @sbom/container.cyclonedx.json

# Query components across SBOMs
curl "https://sbom-db.example.com/api/v1/components?name=requests&version=2.28.0"
```

## Compliance and Reporting

### Government Requirements

#### Executive Order 14028 (US)
- **Requirement**: SBOM for critical software
- **Format**: SPDX or CycloneDX
- **Content**: Comprehensive component inventory
- **Distribution**: Available to customers

#### EU Cyber Resilience Act
- **Requirement**: Software transparency
- **Format**: Machine-readable format
- **Content**: Security-relevant components
- **Maintenance**: Keep updated with patches

### Industry Standards

#### NIST SSDF (Secure Software Development Framework)
- **Practice**: Maintain software inventory
- **Implementation**: Automated SBOM generation
- **Verification**: Regular SBOM validation
- **Distribution**: Provide to stakeholders

#### ISO/IEC 27001
- **Control**: Asset management
- **Implementation**: SBOM as asset inventory
- **Auditing**: Regular SBOM reviews
- **Evidence**: SBOM generation logs

## Automation Scripts

### Complete SBOM Pipeline

```bash
#!/bin/bash
# scripts/generate-sbom.sh

set -euo pipefail

SBOM_DIR="sbom"
IMAGE_NAME="${1:-docker-optimizer:latest}"
FORMATS="${2:-spdx cyclonedx}"

# Create output directory
mkdir -p "$SBOM_DIR"

echo "Generating SBOM for $IMAGE_NAME..."

# Python dependencies
if command -v cyclonedx-py >/dev/null 2>&1; then
    echo "Generating Python dependencies SBOM..."
    cyclonedx-py --format json --output "$SBOM_DIR/python-deps.cyclonedx.json"
else
    echo "Warning: cyclonedx-py not installed, skipping Python SBOM"
fi

# Container image analysis
if command -v syft >/dev/null 2>&1; then
    echo "Generating container image SBOM..."
    for format in $FORMATS; do
        case $format in
            spdx)
                syft "$IMAGE_NAME" -o spdx-json > "$SBOM_DIR/container.spdx.json"
                ;;
            cyclonedx)
                syft "$IMAGE_NAME" -o cyclonedx-json > "$SBOM_DIR/container.cyclonedx.json"
                ;;
        esac
    done
else
    echo "Warning: syft not installed, skipping container SBOM"
fi

# Source code analysis
if command -v cdxgen >/dev/null 2>&1; then
    echo "Generating source code SBOM..."
    npx @cyclonedx/cdxgen -t python -o "$SBOM_DIR/source.cyclonedx.json" .
else
    echo "Warning: cdxgen not available, skipping source SBOM"
fi

# Generate summary
echo "SBOM Generation Summary:"
echo "========================"
for file in "$SBOM_DIR"/*.json; do
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        components=$(jq '.components | length' "$file" 2>/dev/null || echo "N/A")
        echo "$(basename "$file"): $components components, ${size} bytes"
    fi
done

echo "SBOM files generated in $SBOM_DIR/"
```

## Best Practices

### 1. Comprehensive Coverage
- Include all runtime dependencies
- Document build-time dependencies
- Track transitive dependencies
- Include system packages

### 2. Regular Updates
- Regenerate SBOMs with each build
- Update SBOMs when dependencies change
- Maintain historical SBOM versions
- Automate SBOM freshness checks

### 3. Quality Assurance
- Validate SBOM format compliance
- Verify component accuracy
- Check license information completeness
- Monitor SBOM generation failures

### 4. Security Integration
- Use SBOMs for vulnerability scanning
- Cross-reference with threat intelligence
- Implement SBOM-based policy enforcement
- Track component provenance

---

## References

- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [NIST SSDF](https://csrc.nist.gov/Projects/ssdf)
- [CISA SBOM Guide](https://www.cisa.gov/sbom)

*For implementation examples and templates, see `docs/examples/sbom/`*