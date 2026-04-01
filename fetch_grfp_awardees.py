#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.cookiejar
import io
import json
import random
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
        "--limit",
        type=int,
        default=0,
        help="Limit parsed rows for quick inspection. Default: 0, which means no limit.",
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
    retries: int = 3,
) -> FetchStatus:
    payload = urllib.parse.urlencode(
        {
            "method": "loadAwardeeList",
            "awardYear": award_year,
            "awardType": award_type,
            "action": "Search",
        }
    ).encode()
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
    request_cookie_header = cookie_header or (f"JSESSIONID={jsessionid}" if jsessionid else None)
    attempts = max(1, retries)
    last_error_message = "No awardee rows were parsed. The page layout may have changed, or the request was rejected by Research.gov."

    for attempt in range(1, attempts + 1):
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

        get_headers = dict(headers)
        if request_cookie_header:
            get_headers["Cookie"] = request_cookie_header

        bootstrap_request = urllib.request.Request(AWARDEE_LIST_REFERER, headers=get_headers)
        with opener.open(bootstrap_request, timeout=timeout):
            pass

        post_headers = dict(headers)
        if cookie_header:
            post_headers["Cookie"] = cookie_header
        elif jsessionid:
            post_headers["Cookie"] = f"JSESSIONID={jsessionid}"

        request = urllib.request.Request(AWARDEE_LIST_URL, data=payload, headers=post_headers, method="POST")
        with opener.open(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset("latin-1")
            html = response.read().decode(charset, errors="replace")

        parser = AwardeeListParser()
        parser.feed(html)
        if parser.awardees:
            return FetchStatus(
                status="available",
                message=(
                    f"Research.gov returned {len(parser.awardees)} parsed rows for {award_year} "
                    f"award type {award_type}."
                ),
                awardees=parser.awardees,
            )
        if NOT_AVAILABLE_MESSAGE in html:
            return FetchStatus(
                status="not_yet_available",
                message=(
                    f"Research.gov reports that the {award_year} awardee list is not yet available "
                    f"for award type {award_type}."
                ),
                awardees=[],
            )

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
    retries: int = 3,
) -> list[Awardee]:
    status = fetch_status(
        award_year=award_year,
        award_type=award_type,
        timeout=timeout,
        jsessionid=jsessionid,
        cookie_header=cookie_header,
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
        retries=args.retries,
    )
    if args.limit:
        awardees = awardees[: args.limit]

    emit_output(render_output(awardees, args.format), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
