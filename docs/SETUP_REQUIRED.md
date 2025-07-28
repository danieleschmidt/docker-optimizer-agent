# Manual Setup Required

## GitHub Repository Settings

### Branch Protection Rules
- **Branch**: main
- **Requirements**: 
  - Require pull request reviews (1 reviewer minimum)
  - Require status checks to pass
  - Require branches to be up to date
  - Restrict pushes to main branch

### Repository Secrets
- `PYPI_API_TOKEN` - For package publishing
- `DOCKER_USERNAME` - Docker Hub username  
- `DOCKER_PASSWORD` - Docker Hub token
- `CODECOV_TOKEN` - Code coverage reporting (optional)

### Repository Settings
- **Topics**: docker, optimization, security, python, cli
- **Description**: "LLM suggests minimal, secure Dockerfiles and explains each change"
- **Homepage**: Link to documentation or demo

## GitHub Actions Workflows

Create the following workflow files manually in `.github/workflows/`:

1. **ci.yml** - Continuous integration pipeline
2. **docker.yml** - Docker image build and publish  
3. **release.yml** - Release automation
4. **security.yml** - Security scanning schedule

See [docs/workflows/README.md](workflows/README.md) for detailed requirements.

## External Integrations

### Recommended Services
- **CodeCov**: Code coverage reporting
- **Dependabot**: Automated dependency updates
- **CodeQL**: Security analysis
- **pre-commit.ci**: Automated code formatting

## Permissions Required

These tasks require admin access to the repository:
- Creating workflow files in `.github/workflows/`
- Setting up branch protection rules
- Configuring repository secrets
- Enabling security features