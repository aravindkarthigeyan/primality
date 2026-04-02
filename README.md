# GRFP Release Watch

This repository is only for watching the NSF GRFP awardee list on Research.gov.

## What It Contains

- `fetch_grfp_awardees.py`: fetches and parses the GRFP awardee list page
- `.github/workflows/grfp-2026-watch.yml`: checks `2026` every 15 minutes on GitHub Actions

## Local Usage

Check an available year:

```bash
python3 fetch_grfp_awardees.py --award-year 2024 --award-type A --limit 5
```

Check the unreleased year directly:

```bash
python3 fetch_grfp_awardees.py --award-year 2026 --award-type A
```

Run the built-in watch mode:

```bash
python3 fetch_grfp_awardees.py --award-year 2026 --award-type A --watch
```

Useful watch options:

```bash
python3 fetch_grfp_awardees.py \
  --award-year 2026 \
  --award-type A \
  --watch \
  --watch-interval-minutes 180 \
  --output grfp-2026.csv
```

## GitHub Actions Watch

The workflow runs at `:07`, `:22`, `:37`, and `:52` every hour.

When Research.gov starts returning rows for `2026`:

- the workflow uploads first-page CSV and JSON artifacts
- the workflow opens one GitHub issue titled `GRFP 2026 award list is live`
- the workflow can send one email if SMTP secrets are configured

Current workflow link:

- [GRFP 2026 Watch](https://github.com/aravindkarthigeyan/primality/actions/workflows/grfp-2026-watch.yml)

### Email Setup

Add these repository secrets in GitHub:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `ALERT_TO_EMAIL`
- `ALERT_FROM_EMAIL` (optional; falls back to `SMTP_USERNAME`)

Typical Gmail values:

- `SMTP_HOST`: `smtp.gmail.com`
- `SMTP_PORT`: `465`
- `SMTP_USERNAME`: your Gmail address
- `SMTP_PASSWORD`: a Gmail app password
- `ALERT_TO_EMAIL`: the address that should receive the alert

The email is only sent when the workflow creates the one-time release issue, so it does not repeat every 15 minutes after the list goes live.

## Notes

- Research.gov currently returns an explicit “not yet available” message for `2026`.
- The parser currently captures the first result page, which is enough for release detection.
- GitHub scheduled workflows can be delayed under load and are automatically disabled after 60 days of no repository activity in public repositories.
