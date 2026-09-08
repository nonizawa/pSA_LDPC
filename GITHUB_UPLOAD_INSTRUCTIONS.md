# GitHub upload instructions

The repository is already public at https://github.com/nonizawa/pSA_LDPC.
Licensing still requires author approval before the formal release.

## 1. Final local checks

```bash
git diff --stat
python3 scripts/build_release_metadata.py
```

Review `PUBLIC_RELEASE_AUDIT.md`, `PUBLIC_PACKAGE_SIZE_AUDIT.md`, and the
generated `CHECKSUMS.sha256` before continuing.

## 2. Use the existing repository

Do not create another repository, run git init, or add another origin remote.
Review the documentation changes before committing.

## 3. Commit and push the metadata update

```bash
git add README.md CITATION.cff LICENSE_RECOMMENDATION.md GITHUB_UPLOAD_INSTRUCTIONS.md PUBLIC_METADATA_CLEANUP.md PUBLIC_PACKAGE_SIZE_AUDIT.md release_manifest.json CHECKSUMS.sha256
git diff --cached --stat
git commit -m "Update public repository metadata"
git -c http.version=HTTP/1.1 -c http.postBuffer=157286400 push origin main
```

Inspect the updated GitHub README. If authentication is requested, use your
personal access token at the password prompt, not your account password.
Never put credentials in commands or repository files.

## 4. Public release

Finalize the software and data licenses, update README and CITATION.cff,
refresh integrity metadata, and commit/push those changes first.
The version in CITATION.cff is prepared for v1.0.0; no tag or release has
been created by this documentation update.

Enable the repository in Zenodo's GitHub settings, then use GitHub Releases
to publish a release with tag v1.0.0 targeting main. A tag push alone is not
the same as publishing a GitHub Release.

After Zenodo processing, verify the archived files and metadata before citing
the version DOI. The existing verified GitHub URL can be cited meanwhile.
