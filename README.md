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

- the workflow uploads full-list CSV, JSON, and school-summary artifacts
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

You can also test the email wiring immediately:

- open the workflow page
- click `Run workflow`
- set `year` to `2025`
- enable the `test_email` checkbox
- run it manually

That sends a test message without opening the real release issue. When the selected year is available, the email includes:

- the full CSV as an attachment
- the full JSON as an attachment
- a school summary text attachment
- highlighted baccalaureate school names in the email body

## Notes

- Research.gov currently returns an explicit “not yet available” message for `2026`.
- The parser now follows pagination to fetch the full result set when reports are generated.
- GitHub scheduled workflows can be delayed under load and are automatically disabled after 60 days of no repository activity in public repositories.
