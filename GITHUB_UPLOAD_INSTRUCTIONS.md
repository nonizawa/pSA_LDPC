# GitHub upload instructions

No repository or remote was created during package preparation. Complete the
human decisions in `PUBLIC_RELEASE_AUDIT.md`, especially licensing and public
contact information, before making the repository public.

## 1. Final local checks

```bash
python scripts/run_validation.py
python scripts/reproduce_figures.py
python scripts/build_release_metadata.py
```

Review `PUBLIC_RELEASE_AUDIT.md`, `PUBLIC_PACKAGE_SIZE_AUDIT.md`, and the
generated `CHECKSUMS.sha256` before continuing.

## 2. Create an empty GitHub repository

Choose the account/organization and repository name manually. Do not initialize
the remote with a README or license if those files have already been finalized
locally.

## 3. Initialize and push this staging directory

```bash
git init
git add .
git status
git commit -m "Prepare manuscript reproducibility package"
git branch -M main
git remote add origin <REPOSITORY_URL>
git push -u origin main
```

Inspect the GitHub file list and README while the repository is private. Run the
secret/path scan again after any final edit.

## 4. Public release

After author approval, make the repository public and create an immutable
release. `v1.0.0` is a reasonable tag for the manuscript-linked archival
snapshot, but the final version remains an author decision.

```bash
git tag -a v1.0.0 -m "Manuscript reproducibility release"
git push origin v1.0.0
```

Do not insert a repository URL or DOI into the manuscript until the public URL
and archived release identifier have been verified.
