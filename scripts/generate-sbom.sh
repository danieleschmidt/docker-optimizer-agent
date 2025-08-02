#!/bin/bash
# Generate Software Bill of Materials (SBOM)
# Usage: ./scripts/generate-sbom.sh [image-name] [formats]

set -euo pipefail

# Configuration
SBOM_DIR="sbom"
IMAGE_NAME="${1:-docker-optimizer:latest}"
FORMATS="${2:-spdx cyclonedx}"
TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date -u +"%Y-%m-%d %H:%M:%S")] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check required tools
check_dependencies() {
    local missing_tools=()
    
    if ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi
    
    if ! command -v docker >/dev/null 2>&1; then
        missing_tools+=("docker")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        echo "Please install missing tools and try again."
        exit 1
    fi
}

# Install SBOM tools if needed
install_sbom_tools() {
    log "Checking SBOM generation tools..."
    
    # Check and install cyclonedx-py
    if ! command -v cyclonedx-py >/dev/null 2>&1; then
        warn "cyclonedx-py not found, installing..."
        pip install cyclonedx-bom || {
            error "Failed to install cyclonedx-bom"
            return 1
        }
    fi
    
    # Check and install syft
    if ! command -v syft >/dev/null 2>&1; then
        warn "syft not found, installing..."
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin || {
            error "Failed to install syft"
            return 1
        }
    fi
    
    # Check cdxgen (optional)
    if ! command -v cdxgen >/dev/null 2>&1 && ! npx @cyclonedx/cdxgen --version >/dev/null 2>&1; then
        warn "cdxgen not available, will skip source code SBOM"
    fi
    
    success "SBOM tools ready"
}

# Create output directory
setup_output_dir() {
    mkdir -p "$SBOM_DIR"
    log "Created output directory: $SBOM_DIR"
}

# Generate Python dependencies SBOM
generate_python_sbom() {
    log "Generating Python dependencies SBOM..."
    
    if command -v cyclonedx-py >/dev/null 2>&1; then
        # Standard dependencies
        cyclonedx-py --format json --output "$SBOM_DIR/python-deps.cyclonedx.json" || {
            error "Failed to generate Python dependencies SBOM"
            return 1
        }
        
        # Include development dependencies
        cyclonedx-py --format json --include-dev --output "$SBOM_DIR/python-dev-deps.cyclonedx.json" || {
            warn "Failed to generate Python dev dependencies SBOM"
        }
        
        success "Python dependencies SBOM generated"
    else
        warn "cyclonedx-py not available, skipping Python SBOM"
    fi
}

# Generate container image SBOM
generate_container_sbom() {
    log "Generating container image SBOM for $IMAGE_NAME..."
    
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        error "Docker image $IMAGE_NAME not found"
        return 1
    fi
    
    if command -v syft >/dev/null 2>&1; then
        for format in $FORMATS; do
            case $format in
                spdx)
                    log "Generating SPDX format..."
                    syft "$IMAGE_NAME" -o spdx-json > "$SBOM_DIR/container.spdx.json" || {
                        error "Failed to generate SPDX SBOM"
                        continue
                    }
                    ;;
                cyclonedx)
                    log "Generating CycloneDX format..."
                    syft "$IMAGE_NAME" -o cyclonedx-json > "$SBOM_DIR/container.cyclonedx.json" || {
                        error "Failed to generate CycloneDX SBOM"
                        continue
                    }
                    ;;
                *)
                    warn "Unknown format: $format"
                    ;;
            esac
        done
        
        success "Container image SBOM generated"
    else
        warn "syft not available, skipping container SBOM"
    fi
}

# Generate source code SBOM
generate_source_sbom() {
    log "Generating source code SBOM..."
    
    if command -v cdxgen >/dev/null 2>&1 || npx @cyclonedx/cdxgen --version >/dev/null 2>&1; then
        if command -v cdxgen >/dev/null 2>&1; then
            cdxgen -t python -o "$SBOM_DIR/source.cyclonedx.json" . || {
                error "Failed to generate source code SBOM with cdxgen"
                return 1
            }
        else
            npx @cyclonedx/cdxgen -t python -o "$SBOM_DIR/source.cyclonedx.json" . || {
                error "Failed to generate source code SBOM with npx cdxgen"
                return 1
            }
        fi
        
        success "Source code SBOM generated"
    else
        warn "cdxgen not available, skipping source code SBOM"
    fi
}

# Validate generated SBOMs
validate_sboms() {
    log "Validating generated SBOMs..."
    
    local validated=0
    local total=0
    
    for sbom_file in "$SBOM_DIR"/*.json; do
        if [[ -f "$sbom_file" ]]; then
            total=$((total + 1))
            log "Validating $(basename "$sbom_file")..."
            
            # Basic JSON validation
            if jq empty "$sbom_file" >/dev/null 2>&1; then
                validated=$((validated + 1))
                
                # Format-specific validation
                if [[ "$sbom_file" == *"cyclonedx"* ]]; then
                    # CycloneDX validation
                    if command -v cyclonedx-py >/dev/null 2>&1; then
                        cyclonedx-py validate --input-file "$sbom_file" >/dev/null 2>&1 && {
                            success "$(basename "$sbom_file") is valid CycloneDX"
                        } || {
                            warn "$(basename "$sbom_file") failed CycloneDX validation"
                        }
                    fi
                elif [[ "$sbom_file" == *"spdx"* ]]; then
                    # SPDX validation (basic structure check)
                    if jq -e '.spdxVersion and .creationInfo and .packages' "$sbom_file" >/dev/null 2>&1; then
                        success "$(basename "$sbom_file") has valid SPDX structure"
                    else
                        warn "$(basename "$sbom_file") may have invalid SPDX structure"
                    fi
                fi
            else
                error "$(basename "$sbom_file") is not valid JSON"
            fi
        fi
    done
    
    log "Validated $validated out of $total SBOM files"
}

# Generate quality report
generate_quality_report() {
    log "Generating SBOM quality report..."
    
    local report_file="$SBOM_DIR/quality-report.txt"
    
    {
        echo "SBOM Quality Report"
        echo "==================="
        echo "Generated: $(date -u)"
        echo "Image: $IMAGE_NAME"
        echo ""
        
        for sbom_file in "$SBOM_DIR"/*.json; do
            if [[ -f "$sbom_file" ]]; then
                echo "File: $(basename "$sbom_file")"
                echo "-----------------------------"
                
                # File size
                local size
                size=$(stat -f%z "$sbom_file" 2>/dev/null || stat -c%s "$sbom_file" 2>/dev/null || echo "unknown")
                echo "Size: $size bytes"
                
                # Component count
                local components
                components=$(jq '.components | length' "$sbom_file" 2>/dev/null || echo "N/A")
                echo "Components: $components"
                
                # License coverage (for CycloneDX)
                if [[ "$sbom_file" == *"cyclonedx"* ]]; then
                    local licensed
                    licensed=$(jq '[.components[] | select(.licenses != null)] | length' "$sbom_file" 2>/dev/null || echo "0")
                    if [[ "$components" != "N/A" && "$components" != "0" ]]; then
                        local coverage=$((licensed * 100 / components))
                        echo "License Coverage: $coverage% ($licensed/$components)"
                    fi
                fi
                
                # Package types (for SPDX)
                if [[ "$sbom_file" == *"spdx"* ]]; then
                    local packages
                    packages=$(jq '.packages | length' "$sbom_file" 2>/dev/null || echo "N/A")
                    echo "Packages: $packages"
                fi
                
                echo ""
            fi
        done
        
        echo "Generation completed at: $(date -u)"
    } > "$report_file"
    
    success "Quality report generated: $report_file"
}

# Create timestamped archive
create_archive() {
    log "Creating timestamped archive..."
    
    local archive_name="sbom-${TIMESTAMP}.tar.gz"
    tar -czf "$archive_name" -C "$SBOM_DIR" . || {
        error "Failed to create archive"
        return 1
    }
    
    success "Archive created: $archive_name"
}

# Main execution
main() {
    log "Starting SBOM generation for $IMAGE_NAME"
    log "Output directory: $SBOM_DIR"
    log "Formats: $FORMATS"
    
    check_dependencies
    install_sbom_tools
    setup_output_dir
    
    # Generate different types of SBOMs
    generate_python_sbom
    generate_container_sbom
    generate_source_sbom
    
    # Validate and report
    validate_sboms
    generate_quality_report
    
    # Create archive
    create_archive
    
    # Final summary
    echo ""
    success "SBOM generation completed successfully!"
    echo ""
    echo "Generated files:"
    for file in "$SBOM_DIR"/*; do
        if [[ -f "$file" ]]; then
            echo "  $(basename "$file")"
        fi
    done
    echo ""
    log "All SBOM files available in: $SBOM_DIR/"
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [image-name] [formats]"
        echo ""
        echo "Arguments:"
        echo "  image-name    Docker image name (default: docker-optimizer:latest)"
        echo "  formats       SBOM formats to generate: spdx, cyclonedx (default: spdx cyclonedx)"
        echo ""
        echo "Examples:"
        echo "  $0                                    # Use defaults"
        echo "  $0 my-image:v1.0                    # Custom image"
        echo "  $0 my-image:v1.0 \"spdx\"             # SPDX format only"
        echo "  $0 my-image:v1.0 \"spdx cyclonedx\"   # Both formats"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac