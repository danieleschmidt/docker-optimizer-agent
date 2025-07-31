# Troubleshooting Guide

## Common Issues

### Installation Problems

#### Missing Dependencies
```bash
# Error: ModuleNotFoundError
pip install docker-optimizer-agent[security]

# Or install all optional dependencies
pip install docker-optimizer-agent[dev,security,trivy]
```

#### Permission Errors
```bash
# Use --user flag for local installation
pip install --user docker-optimizer-agent

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Runtime Issues

#### Docker Not Available
```bash
# Ensure Docker is running
docker --version
sudo systemctl start docker  # Linux
```

#### Trivy Scanner Issues
```bash
# Install Trivy manually
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Or use Docker image
docker run --rm -v "$PWD":/work aquasecurity/trivy:latest
```

### Performance Issues

#### Slow Optimization
- **Cause**: Large Dockerfiles or slow network
- **Solution**: Use `--performance` flag for caching
- **Alternative**: Process in batches with `--batch`

#### High Memory Usage
- **Monitoring**: `docker stats` during processing
- **Optimization**: Use streaming mode for large files
- **Limits**: Set container memory limits

### CI/CD Integration Issues

#### GitHub Actions Failures
```yaml
# Add timeout and retry logic
- name: Optimize Dockerfiles
  run: docker-optimizer --dockerfile Dockerfile --timeout 300
  timeout-minutes: 10
```

#### Pre-commit Hook Failures
```bash
# Skip specific hooks temporarily
SKIP=dockerfile-optimization git commit -m "fix: temporary skip"

# Update hook configuration
pre-commit autoupdate
```

## Debug Mode

### Enable Verbose Logging
```bash
# CLI verbose mode
docker-optimizer --dockerfile Dockerfile --verbose

# Python API debug
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler

# Profile optimization
python -m memory_profiler -c "from docker_optimizer import DockerfileOptimizer; DockerfileOptimizer().optimize_dockerfile(open('Dockerfile').read())"
```

### Performance Profiling
```bash
# Install profiling tools
pip install py-spy

# Profile running process
py-spy record -o profile.svg -- docker-optimizer --dockerfile Dockerfile
```

## Support Channels

### Issue Reporting
1. **GitHub Issues**: [Report bugs and feature requests](https://github.com/danieleschmidt/docker-optimizer-agent/issues)
2. **Security Issues**: Email security@terragonlabs.com
3. **Performance Issues**: Use the performance issue template

### Before Reporting
- [ ] Check this troubleshooting guide
- [ ] Search existing GitHub issues
- [ ] Test with minimal Dockerfile
- [ ] Include version information (`docker-optimizer --version`)
- [ ] Provide complete error messages

### Information to Include
```bash
# System information
docker-optimizer --version
python --version
docker --version

# Environment details
pip list | grep docker-optimizer
echo $DOCKER_HOST
```

## Known Limitations

### Platform Compatibility
- **Windows**: Some shell commands may require WSL
- **ARM64**: Limited base image optimization
- **Legacy Docker**: Requires Docker 20.10+

### External Dependencies
- **Trivy**: Requires internet access for vulnerability database
- **Registry APIs**: Rate limiting may affect optimization
- **Base Images**: Some optimizations require image availability