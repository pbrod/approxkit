# ApproxKit Release Workflow

> ApproxKit uses `setuptools-scm` for versioning.
>
> Package versions are derived automatically from Git tags. Do not manually edit
> `__version__` when preparing a release.

---

# Workflows

The repository uses three GitHub Actions workflows:

| Workflow | Purpose | Trigger |
|----------|----------|----------|
| `tests.yml` | Run tests and packaging checks | Push and Pull Request |
| `test_publish.yml` | Upload package to TestPyPI | Manual |
| `publish.yml` | Upload package to PyPI | GitHub Release |

---

# 1. Development

While developing ApproxKit:

```bash
pytest --pyargs approxkit
```

or

```python
import approxkit

approxkit.test()
```

Commit changes normally:

```bash
git add .
git commit -m "..."
git push
```

Every push and pull request automatically triggers:

```text
.github/workflows/tests.yml
```

which performs:

- installation
- test execution
- package build
- wheel validation

---

# 2. Continuous Integration

The CI workflow (`tests.yml`) runs:

```bash
pytest --pyargs approxkit
```

and also verifies:

```bash
python -m build
twine check dist/*
```

The package must pass CI before any release.

---

# 3. TestPyPI Release

Before publishing a new release to PyPI, test the package on TestPyPI.

## Trigger

Go to:

```text
GitHub
→ Actions
→ Publish TestPyPI
→ Run workflow
```

This executes:

```text
.github/workflows/test_publish.yml
```

which:

1. Builds the package.
2. Creates wheel and source distributions.
3. Uploads them to TestPyPI.

## Verify Installation

Install from TestPyPI:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  approxkit
```

Verify:

```python
import approxkit

print(approxkit.__version__)
```

Run a few quick checks:

```python
from approxkit import chebyshev_nodes

print(chebyshev_nodes(4))
```

If testing support is desired:

```bash
pip install pytest
```

and then:

```python
approxkit.test("-q")
```

If everything works correctly, proceed with the real release.

---

# 4. Create Release

ApproxKit uses `setuptools-scm`.

Versions are determined automatically from Git tags.

Create and push a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Verify:

```bash
git tag
```

contains the new release tag.

---

# 5. Publish to PyPI

Create a GitHub Release:

```text
GitHub
→ Releases
→ Draft new release
```

Select:

```text
Tag: vX.Y.Z
```

Add release notes and click:

```text
Publish release
```

Publishing the GitHub Release automatically triggers:

```text
.github/workflows/publish.yml
```

which:

1. Builds the package.
2. Creates wheel and source distributions.
3. Uploads them to PyPI.

---

# 6. Verify PyPI Installation

Install the released package:

```bash
pip install approxkit
```

Verify:

```python
import approxkit

print(approxkit.__version__)
```

Run a basic functionality check:

```python
from approxkit import chebyshev_nodes

print(chebyshev_nodes(4))
```

If pytest is installed:

```python
approxkit.test("-q")
```

---

# Release Checklist

Before a release:

- [ ] All tests pass locally.
- [ ] CI passes on GitHub.
- [ ] TestPyPI upload succeeds.
- [ ] TestPyPI installation verified.
- [ ] Release tag created.
- [ ] GitHub Release published.

After release:

- [ ] Verify installation from PyPI.
- [ ] Verify package version.
- [ ] Verify public APIs work.
- [ ] Verify `approxkit.test()` works (if pytest is installed).

---

# Workflow Summary

```text
Development
    │
    ▼
git push
    │
    ▼
tests.yml
    │
    ▼
Run test_publish.yml manually
    │
    ▼
Upload to TestPyPI
    │
    ▼
Test installation
    │
    ▼
Create git tag
    │
    ▼
Create GitHub Release
    │
    ▼
publish.yml
    │
    ▼
Upload to PyPI
    │
    ▼
pip install approxkit
```