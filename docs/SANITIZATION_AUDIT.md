# Sanitization audit

Scope: every staged text/source/data file, excluding the checksum manifest.

## Private-path and host scan

- No personal absolute home-directory path remains.
- No cloud-sync directory path or author account username remains.
- No source-host name, private IP address, shell history, or private repository
  URL remains.
- Source locations in copied scientific records use the neutral relative label
  `source_project/` where a provenance field had originally stored a path.
- The test suite constructs forbidden path tokens from string fragments so the
  negative-test source itself does not contain a leaked absolute path.

## Credential-shaped value scan

Patterns checked include `TOKEN`, `API_KEY`, `PASSWORD`, `SECRET`, `Bearer`,
`sk-`, `ghp_`, `github_pat_`, `AWS_`, `PRIVATE KEY`, `BEGIN RSA`, and
`BEGIN OPENSSH`, case-insensitively where appropriate. No credential-shaped
value was found.

The scan produced only documented false positives: the ordinary phrase
"secret/path scan" in upload instructions, a local Python variable named
`token` in a negative-path test, and dotted compiler/Python version numbers that
match a broad IPv4-like pattern. None is a credential or network address.

## Excluded artifacts

`.DS_Store`, `__pycache__`, `*.pyc`, notebook checkpoints, `.env`, temporary
smoke outputs, and run logs are absent. Generated PNG smoke artifacts are not
included; current reference and regenerated PDF figures are retained.

The corresponding-author email remains the explicit placeholder
`<PUBLIC_CONTACT_EMAIL>` pending author selection.
