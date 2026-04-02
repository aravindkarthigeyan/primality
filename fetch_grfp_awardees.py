#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import http.cookiejar
import io
import json
import math
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path


AWARDEE_LIST_URL = "https://www.research.gov/grfp/AwardeeList.do"
AWARDEE_LIST_REFERER = "https://www.research.gov/grfp/AwardeeList.do?method=loadAwardeeList"
RESULT_HEADERS = ("Name", "Baccalaureate Institution", "Field of Study", "Current Institution")
CSV_HEADERS = ("name", "baccalaureate_institution", "field_of_study", "current_institution")
NOT_AVAILABLE_MESSAGE = "is not yet available"
RESULT_COUNT_RE = re.compile(r"([0-9,]+)\s+Applicants found,\s+displaying\s+([0-9,]+)\s+to\s+([0-9,]+)", re.I)
DEFAULT_HIGHLIGHT_SCHOOLS = ("University of Texas at Austin",)
SCHOOL_ALIASES = {
    "university of texas at austin": {
        "university of texas at austin",
        "the university of texas at austin",
        "ut austin",
        "u t austin",
    }
}


@dataclass
class Awardee:
    name: str
    baccalaureate_institution: str
    field_of_study: str
    current_institution: str


@dataclass
class FetchStatus:
    status: str
    message: str
    awardees: list[Awardee]


@dataclass
class HighlightSchoolReport:
    requested_label: str
    matched_institutions: list[str]
    names: list[str]


class AwardeeListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_row = False
        self._in_cell = False
        self._current_cell: list[str] = []
        self._current_row: list[str] = []
        self._collect_rows = False
        self._finished = False
        self.awardees: list[Awardee] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._finished:
            return
        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._finished:
            return
        if tag in {"td", "th"} and self._in_cell:
            self._current_row.append(self._normalize("".join(self._current_cell)))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self._consume_row(self._current_row)
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell and not self._finished:
            self._current_cell.append(data)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split())

    def _consume_row(self, row: list[str]) -> None:
        if not any(row):
            return
        if tuple(row) == RESULT_HEADERS:
            self._collect_rows = True
            return
        if not self._collect_rows:
            return
        if len(row) != len(CSV_HEADERS):
            if self.awardees:
                self._finished = True
            return
        self.awardees.append(Awardee(*row))


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def parse_result_window(html: str) -> tuple[int, int, int] | None:
    match = RESULT_COUNT_RE.search(html)
    if not match:
        return None
    total, start, end = (int(group.replace(",", "")) for group in match.groups())
    return total, start, end


def parse_highlight_schools(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_HIGHLIGHT_SCHOOLS)
    schools = [chunk.strip() for chunk in re.split(r"[,\n;]+", value) if chunk.strip()]
    return schools or list(DEFAULT_HIGHLIGHT_SCHOOLS)


def canonical_school_aliases(label: str) -> set[str]:
    normalized = normalize_text(label)
    aliases = {normalized}
    aliases.update(SCHOOL_ALIASES.get(normalized, set()))
    return aliases


def institution_matches_requested_school(institution: str, requested_label: str) -> bool:
    normalized_institution = normalize_text(institution)
    aliases = canonical_school_aliases(requested_label)
    if normalized_institution in aliases:
        return True
    return any(alias in normalized_institution or normalized_institution in alias for alias in aliases if alias)


def build_highlight_school_reports(
    awardees: list[Awardee], highlight_schools: list[str] | None = None
) -> list[HighlightSchoolReport]:
    reports = []
    for requested_label in (highlight_schools or list(DEFAULT_HIGHLIGHT_SCHOOLS)):
        matched = [awardee for awardee in awardees if institution_matches_requested_school(awardee.baccalaureate_institution, requested_label)]
        matched_institutions = sorted({awardee.baccalaureate_institution for awardee in matched})
        reports.append(
            HighlightSchoolReport(
                requested_label=requested_label,
                matched_institutions=matched_institutions,
                names=[awardee.name for awardee in matched],
            )
        )
    return reports


def top_baccalaureate_institutions(awardees: list[Awardee], *, limit: int = 10) -> list[tuple[str, int]]:
    counts = Counter(awardee.baccalaureate_institution for awardee in awardees if awardee.baccalaureate_institution)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return ranked[:limit]


def render_school_summary_text(awardees: list[Awardee], highlight_schools: list[str] | None = None) -> str:
    lines = [
        f"Total parsed awardees: {len(awardees)}",
        "",
        "Top baccalaureate institutions:",
    ]
    for institution, count in top_baccalaureate_institutions(awardees, limit=10):
        lines.append(f"- {institution}: {count}")

    for report in build_highlight_school_reports(awardees, highlight_schools):
        lines.extend(
            [
                "",
                f"{report.requested_label}: {len(report.names)}",
            ]
        )
        if report.matched_institutions:
            lines.append("Matched institution labels:")
            for institution in report.matched_institutions:
                lines.append(f"- {institution}")
        if report.names:
            lines.append("Names:")
            for name in report.names:
                lines.append(f"- {name}")
        else:
            lines.append("Names:")
            lines.append("- none")
    return "\n".join(lines) + "\n"


def build_request_headers(request_cookie_header: str | None) -> tuple[dict[str, str], dict[str, str]]:
    headers = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
            "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.research.gov",
        "Pragma": "no-cache",
        "Referer": AWARDEE_LIST_REFERER,
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
    }
    get_headers = dict(headers)
    if request_cookie_header:
        get_headers["Cookie"] = request_cookie_header
    return headers, get_headers


def decode_response(response: object) -> str:
    charset = response.headers.get_content_charset("latin-1")  # type: ignore[attr-defined]
    return response.read().decode(charset, errors="replace")  # type: ignore[attr-defined]


def fetch_available_awardees(
    *,
    opener: urllib.request.OpenerDirector,
    headers: dict[str, str],
    award_year: str,
    award_type: str,
    timeout: float,
    all_pages: bool,
) -> list[Awardee]:
    post_headers = dict(headers)
    request = urllib.request.Request(
        AWARDEE_LIST_URL,
        data=urllib.parse.urlencode(
            {
                "method": "loadAwardeeList",
                "awardYear": award_year,
                "awardType": award_type,
                "action": "Search",
            }
        ).encode(),
        headers=post_headers,
        method="POST",
    )
    with opener.open(request, timeout=timeout) as response:
        first_html = decode_response(response)

    parser = AwardeeListParser()
    parser.feed(first_html)
    if not parser.awardees:
        if NOT_AVAILABLE_MESSAGE in first_html:
            raise RuntimeError(
                f"Research.gov reports that the {award_year} awardee list is not yet available for award type {award_type}."
            )
        raise RuntimeError("No awardee rows were parsed. The page layout may have changed, or the request was rejected by Research.gov.")

    awardees = list(parser.awardees)
    if not all_pages:
        return awardees

    result_window = parse_result_window(first_html)
    if not result_window:
        return awardees

    total_count, start, end = result_window
    page_size = max(1, end - start + 1)
    total_pages = max(1, math.ceil(total_count / page_size))

    for page_number in range(2, total_pages + 1):
        page_request = urllib.request.Request(
            f"{AWARDEE_LIST_URL}?method=sort&page={page_number}",
            headers=headers,
        )
        with opener.open(page_request, timeout=timeout) as response:
            html = decode_response(response)
        page_parser = AwardeeListParser()
        page_parser.feed(html)
        if not page_parser.awardees:
            raise RuntimeError(f"Research.gov returned no rows for page {page_number} while fetching the full list.")
        awardees.extend(page_parser.awardees)

    return awardees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and parse NSF GRFP awardee or honorable mention results from Research.gov. "
            "The script first loads the live form to establish any required session cookies."
        )
    )
    parser.add_argument("--award-year", required=True, help="Award year to request, for example 2025.")
    parser.add_argument(
        "--award-type",
        default="A",
        choices=("A", "H"),
        help="`A` for award offers, `H` for honorable mentions. Default: A.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=("csv", "json"),
        help="Output format. Default: csv.",
    )
    parser.add_argument("--output", type=Path, help="Optional output path. Defaults to stdout.")
    parser.add_argument(
        "--school-summary-output",
        type=Path,
        help="Optional path for a plain-text baccalaureate school summary report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit parsed rows for quick inspection. Default: 0, which means no limit.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Follow Research.gov pagination and fetch the entire result set instead of only the first page.",
    )
    parser.add_argument(
        "--highlight-baccalaureate-schools",
        default="University of Texas at Austin",
        help="Comma-separated school names to highlight in the school summary. Default: University of Texas at Austin.",
    )
    parser.add_argument(
        "--jsessionid",
        help="Optional Research.gov JSESSIONID cookie value to reuse instead of the script-created session.",
    )
    parser.add_argument(
        "--cookie-header",
        help="Optional raw Cookie header. Overrides --jsessionid when both are set.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for intermittent empty responses from Research.gov. Default: 3.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll Research.gov until the requested year becomes available.",
    )
    parser.add_argument(
        "--watch-interval-minutes",
        type=float,
        default=240.0,
        help="Minutes between checks in --watch mode. Default: 240.",
    )
    parser.add_argument(
        "--watch-jitter-seconds",
        type=float,
        default=120.0,
        help="Add up to this many random seconds between watch checks. Default: 120.",
    )
    parser.add_argument(
        "--watch-max-checks",
        type=int,
        default=0,
        help="Maximum watch checks before exiting. Default: 0, which means run until released or interrupted.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds. Default: 30.")
    return parser.parse_args()


def fetch_status(
    *,
    award_year: str,
    award_type: str,
    timeout: float,
    jsessionid: str | None,
    cookie_header: str | None,
    all_pages: bool = False,
    retries: int = 3,
) -> FetchStatus:
    request_cookie_header = cookie_header or (f"JSESSIONID={jsessionid}" if jsessionid else None)
    attempts = max(1, retries)
    last_error_message = "No awardee rows were parsed. The page layout may have changed, or the request was rejected by Research.gov."

    for attempt in range(1, attempts + 1):
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        headers, get_headers = build_request_headers(request_cookie_header)

        bootstrap_request = urllib.request.Request(AWARDEE_LIST_REFERER, headers=get_headers)
        with opener.open(bootstrap_request, timeout=timeout):
            pass

        try:
            awardees = fetch_available_awardees(
                opener=opener,
                headers=headers,
                award_year=award_year,
                award_type=award_type,
                timeout=timeout,
                all_pages=all_pages,
            )
            scope = "across the full result set" if all_pages else "on the first page"
            return FetchStatus(
                status="available",
                message=(
                    f"Research.gov returned {len(awardees)} parsed rows for {award_year} "
                    f"award type {award_type} {scope}."
                ),
                awardees=awardees,
            )
        except RuntimeError as exc:
            if NOT_AVAILABLE_MESSAGE in str(exc):
                return FetchStatus(
                    status="not_yet_available",
                    message=str(exc),
                    awardees=[],
                )
            last_error_message = str(exc)

        if attempt < attempts:
            time.sleep(0.5 * attempt)

    return FetchStatus(status="error", message=last_error_message, awardees=[])


def fetch_awardees(
    *,
    award_year: str,
    award_type: str,
    timeout: float,
    jsessionid: str | None,
    cookie_header: str | None,
    all_pages: bool = False,
    retries: int = 3,
) -> list[Awardee]:
    status = fetch_status(
        award_year=award_year,
        award_type=award_type,
        timeout=timeout,
        jsessionid=jsessionid,
        cookie_header=cookie_header,
        all_pages=all_pages,
        retries=retries,
    )
    if status.status != "available":
        raise RuntimeError(status.message)
    return status.awardees


def render_csv(awardees: list[Awardee]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_HEADERS)
    writer.writeheader()
    writer.writerows(asdict(awardee) for awardee in awardees)
    return buffer.getvalue()


def render_json(awardees: list[Awardee]) -> str:
    return json.dumps([asdict(awardee) for awardee in awardees], indent=2)


def format_timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def render_output(awardees: list[Awardee], output_format: str) -> str:
    if output_format == "json":
        return render_json(awardees)
    return render_csv(awardees)


def emit_output(output: str, output_path: Path | None) -> None:
    if output_path:
        output_path.write_text(output, encoding="utf-8")
        print(f"[{format_timestamp()}] wrote results to {output_path}", file=sys.stderr, flush=True)
        return
    sys.stdout.write(output)
    if not output.endswith("\n"):
        sys.stdout.write("\n")


def watch_for_release(args: argparse.Namespace) -> int:
    last_status: str | None = None
    check_number = 0
    interval_seconds = args.watch_interval_minutes * 60.0

    while True:
        check_number += 1
        status = fetch_status(
            award_year=args.award_year,
            award_type=args.award_type,
            timeout=args.timeout,
            jsessionid=args.jsessionid,
            cookie_header=args.cookie_header,
            all_pages=False,
            retries=args.retries,
        )
        timestamp = format_timestamp()

        if status.status == "available":
            awardees = status.awardees
            if args.limit:
                awardees = awardees[: args.limit]
            print(
                f"[{timestamp}] released on check {check_number}: {len(awardees)} row(s) "
                f"captured for {args.award_year} {args.award_type}",
                file=sys.stderr,
                flush=True,
            )
            emit_output(render_output(awardees, args.format), args.output)
            return 0

        if status.status == "not_yet_available":
            if status.status != last_status:
                print(f"[{timestamp}] {status.message}", file=sys.stderr, flush=True)
            else:
                print(
                    f"[{timestamp}] still unavailable for {args.award_year} {args.award_type}; "
                    f"checked {check_number} time(s)",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            print(f"[{timestamp}] check {check_number} failed: {status.message}", file=sys.stderr, flush=True)

        last_status = status.status
        if args.watch_max_checks > 0 and check_number >= args.watch_max_checks:
            print(f"[{timestamp}] stopping after {check_number} checks", file=sys.stderr, flush=True)
            return 2

        sleep_seconds = max(0.0, interval_seconds)
        if args.watch_jitter_seconds > 0:
            sleep_seconds += random.uniform(0.0, args.watch_jitter_seconds)
        print(
            f"[{timestamp}] next check in about {sleep_seconds / 60.0:.1f} minutes",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(sleep_seconds)


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be greater than or equal to 0")
    if args.watch_interval_minutes < 0:
        raise SystemExit("--watch-interval-minutes must be greater than or equal to 0")
    if args.watch_jitter_seconds < 0:
        raise SystemExit("--watch-jitter-seconds must be greater than or equal to 0")
    if args.watch_max_checks < 0:
        raise SystemExit("--watch-max-checks must be greater than or equal to 0")

    if args.watch:
        return watch_for_release(args)

    awardees = fetch_awardees(
        award_year=args.award_year,
        award_type=args.award_type,
        timeout=args.timeout,
        jsessionid=args.jsessionid,
        cookie_header=args.cookie_header,
        all_pages=args.all_pages,
        retries=args.retries,
    )
    if args.limit:
        awardees = awardees[: args.limit]

    emit_output(render_output(awardees, args.format), args.output)
    if args.school_summary_output:
        args.school_summary_output.write_text(
            render_school_summary_text(
                awardees,
                parse_highlight_schools(args.highlight_baccalaureate_schools),
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
