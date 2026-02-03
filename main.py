import argparse
import datetime as dt
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


LIST_URL_DEFAULT = "https://jobs.inria.fr/public/classic/fr/offres"
DETAIL_PATH_RE = re.compile(r"^/public/classic/(fr|en)/offres/(\d{4}-\d+)$")


@dataclass
class OfferListItem:
    reference: str
    title: Optional[str]
    url: str
    ville: Optional[str]
    equipe: Optional[str]
    date_limite: Optional[str]
    list_page_url: Optional[str]


def utc_now_iso() -> str:
    ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")


def norm_url(u: str) -> str:
    p = urlparse(u)
    p = p._replace(fragment="")
    return urlunparse(p)


def mk_session(user_agent: str, timeout_s: float) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
            "Connection": "keep-alive",
        }
    )
    orig = s.request

    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout_s
        return orig(method, url, **kwargs)

    s.request = wrapped
    return s


def fetch_html(s: requests.Session, url: str, retries: int, backoff_base_s: float) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            r = s.get(url, allow_redirects=True)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except Exception as e:
            last_err = e
            if i == retries:
                break
            time.sleep(backoff_base_s * (2**i) + random.random() * 0.25)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def rx_field(text: str, keys: List[str]) -> Optional[str]:
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s*:\s*([^•\n\r]+)", text, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()
            v = re.sub(r"\s+", " ", v)
            return v
    return None


def extract_segment_text(h2_tag) -> str:
    parts = []
    cur = h2_tag
    for _ in range(40):
        cur = cur.find_next_sibling()
        if cur is None:
            break
        if getattr(cur, "name", None) == "h2":
            break
        t = cur.get_text(" ", strip=True)
        if t:
            parts.append(t)
    return " ".join(parts)


def parse_offers_from_list(html: str, page_url: str) -> List[OfferListItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: Dict[str, OfferListItem] = {}

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        m = DETAIL_PATH_RE.match(href)
        if not m:
            continue
        ref = m.group(2)
        abs_url = urljoin(page_url, href)
        title = a.get_text(" ", strip=True) or None
        h2 = a.find_parent("h2")
        seg_text = extract_segment_text(h2) if h2 else a.find_parent().get_text(" ", strip=True)
        ville = rx_field(seg_text, ["Ville", "Town/city"])
        equipe = rx_field(seg_text, ["Équipe Inria", "Inria Team"])
        date_limite = rx_field(seg_text, ["Date limite pour postuler", "Deadline to apply"])

        if ref not in items:
            items[ref] = OfferListItem(
                reference=ref,
                title=title,
                url=abs_url,
                ville=ville,
                equipe=equipe,
                date_limite=date_limite,
                list_page_url=page_url,
            )

    return [items[k] for k in sorted(items.keys())]


def discover_list_pages(html: str, page_url: str, list_path_prefix: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: Set[str] = set()

    def add(u: str):
        if not u:
            return
        abs_u = norm_url(urljoin(page_url, u))
        path = urlparse(abs_u).path or ""
        if DETAIL_PATH_RE.match(path):
            return
        if path.startswith(list_path_prefix):
            candidates.add(abs_u)

    for a in soup.find_all("a", href=True):
        add(a.get("href"))

    for tag in soup.find_all(True):
        for k, v in list(tag.attrs.items()):
            if isinstance(v, str):
                lk = str(k).lower()
                if lk.startswith("data-") and ("/offres" in v or "paginate" in v or "page=" in v or "start=" in v or "offset=" in v):
                    add(v)

    candidates.discard(norm_url(page_url))

    def score(u: str) -> Tuple[int, str]:
        s = 0
        if "paginate" in u:
            s += 3
        if "page=" in u or "p=" in u:
            s += 2
        if "start=" in u or "offset=" in u:
            s += 1
        return (-s, u)

    return sorted(candidates, key=score)


def parse_detail_fields(detail_html: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(detail_html, "html.parser")
    text = soup.get_text("\n", strip=True)
    return {
        "type_contrat": rx_field(text, ["Type de contrat", "Contract type"]),
        "fonction": rx_field(text, ["Fonction", "Function"]),
        "niveau_diplome": rx_field(text, ["Niveau de diplôme exigé", "Required level of education"]),
    }


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            start_url TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS offers (
            reference TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            ville TEXT,
            equipe TEXT,
            date_limite TEXT,
            list_page_url TEXT,
            type_contrat TEXT,
            fonction TEXT,
            niveau_diplome TEXT,
            detail_html TEXT,
            first_seen_at_utc TEXT NOT NULL,
            last_seen_at_utc TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_offers (
            run_id INTEGER NOT NULL,
            reference TEXT NOT NULL,
            seen_at_utc TEXT NOT NULL,
            is_new INTEGER NOT NULL,
            PRIMARY KEY (run_id, reference),
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (reference) REFERENCES offers(reference)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_first_seen ON offers(first_seen_at_utc);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_last_seen ON offers(last_seen_at_utc);")
    conn.commit()


def start_run(conn: sqlite3.Connection, start_url: str) -> Tuple[int, str]:
    ts = utc_now_iso()
    cur = conn.execute("INSERT INTO runs(started_at_utc, start_url) VALUES (?, ?)", (ts, start_url))
    conn.commit()
    return int(cur.lastrowid), ts


def finish_run(conn: sqlite3.Connection, run_id: int) -> str:
    ts = utc_now_iso()
    conn.execute("UPDATE runs SET finished_at_utc=? WHERE run_id=?", (ts, run_id))
    conn.commit()
    return ts


def offer_exists(conn: sqlite3.Connection, reference: str) -> bool:
    row = conn.execute("SELECT 1 FROM offers WHERE reference=? LIMIT 1", (reference,)).fetchone()
    return row is not None


def insert_offer(conn: sqlite3.Connection, item: OfferListItem, now_ts: str, detail_fields: Dict[str, Optional[str]], detail_html: Optional[str]) -> None:
    conn.execute(
        """
        INSERT INTO offers(
            reference, url, title, ville, equipe, date_limite, list_page_url,
            type_contrat, fonction, niveau_diplome, detail_html,
            first_seen_at_utc, last_seen_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.reference,
            item.url,
            item.title,
            item.ville,
            item.equipe,
            item.date_limite,
            item.list_page_url,
            detail_fields.get("type_contrat"),
            detail_fields.get("fonction"),
            detail_fields.get("niveau_diplome"),
            detail_html,
            now_ts,
            now_ts,
        ),
    )


def update_offer(conn: sqlite3.Connection, item: OfferListItem, now_ts: str, detail_fields: Dict[str, Optional[str]], detail_html: Optional[str]) -> None:
    conn.execute(
        """
        UPDATE offers SET
            url=?,
            title=?,
            ville=?,
            equipe=?,
            date_limite=?,
            list_page_url=?,
            type_contrat=COALESCE(?, type_contrat),
            fonction=COALESCE(?, fonction),
            niveau_diplome=COALESCE(?, niveau_diplome),
            detail_html=COALESCE(?, detail_html),
            last_seen_at_utc=?
        WHERE reference=?
        """,
        (
            item.url,
            item.title,
            item.ville,
            item.equipe,
            item.date_limite,
            item.list_page_url,
            detail_fields.get("type_contrat"),
            detail_fields.get("fonction"),
            detail_fields.get("niveau_diplome"),
            detail_html,
            now_ts,
            item.reference,
        ),
    )


def link_run_offer(conn: sqlite3.Connection, run_id: int, reference: str, seen_at: str, is_new: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO run_offers(run_id, reference, seen_at_utc, is_new) VALUES (?, ?, ?, ?)",
        (run_id, reference, seen_at, is_new),
    )


def _norm_title(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def _is_excluded_title(title: Optional[str], excluded_tokens_casefold: List[str]) -> bool:
    t = _norm_title(title)
    if not t:
        return False
    return any(tok in t for tok in excluded_tokens_casefold)


def crawl(
    start_url: str,
    db_path: str,
    with_details: bool,
    max_pages: int,
    delay_s: float,
    timeout_s: float,
    retries: int,
    backoff_base_s: float,
    user_agent: str,
    excluded_title_tokens: List[str],
) -> None:
    start_url = norm_url(start_url)
    list_prefix = (urlparse(start_url).path or "").rstrip("/")
    excluded_tokens_casefold = [re.sub(r"\s+", " ", t.strip()).casefold() for t in excluded_title_tokens if t.strip()]

    s = mk_session(user_agent=user_agent, timeout_s=timeout_s)
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        run_id, run_started = start_run(conn, start_url)

        q: List[str] = [start_url]
        seen_pages: Set[str] = set()
        seen_refs_in_run: Set[str] = set()

        pages_done = 0
        new_count = 0
        total_seen = 0
        excluded_count = 0

        while q and pages_done < max_pages:
            page_url = norm_url(q.pop(0))
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)

            html = fetch_html(s, page_url, retries=retries, backoff_base_s=backoff_base_s)
            pages_done += 1

            items = parse_offers_from_list(html, page_url)
            for item in items:
                if item.reference in seen_refs_in_run:
                    continue
                seen_refs_in_run.add(item.reference)

                if _is_excluded_title(item.title, excluded_tokens_casefold):
                    excluded_count += 1
                    continue

                total_seen += 1

                now_ts = utc_now_iso()
                is_new = 0
                detail_html = None
                detail_fields = {"type_contrat": None, "fonction": None, "niveau_diplome": None}

                exists = offer_exists(conn, item.reference)
                if with_details:
                    time.sleep(delay_s)
                    detail_html = fetch_html(s, item.url, retries=retries, backoff_base_s=backoff_base_s)
                    detail_fields = parse_detail_fields(detail_html)

                if not exists:
                    is_new = 1
                    new_count += 1
                    print(f"NEW {item.reference} | {item.title or ''} | {item.url}", flush=True)
                    insert_offer(conn, item, now_ts, detail_fields, detail_html)
                else:
                    update_offer(conn, item, now_ts, detail_fields, detail_html)

                link_run_offer(conn, run_id, item.reference, now_ts, is_new)

            conn.commit()

            next_pages = discover_list_pages(html, page_url, list_prefix)
            for u in next_pages:
                if u not in seen_pages:
                    q.append(u)

            time.sleep(delay_s)

        run_finished = finish_run(conn, run_id)
        print(
            f"Run {run_id} done. started={run_started} finished={run_finished} pages={pages_done} kept_seen={total_seen} excluded={excluded_count} new={new_count} db={db_path}",
            flush=True,
        )
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-url", default=LIST_URL_DEFAULT)
    p.add_argument("--db", default="inria_jobs.sqlite")
    p.add_argument("--with-details", action="store_true")
    p.add_argument("--max-pages", type=int, default=50)
    p.add_argument("--delay", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=25.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=0.8)
    p.add_argument("--user-agent", default="Mozilla/5.0 (compatible; inria-jobs-scraper/1.0)")
    p.add_argument(
        "--exclude-title",
        action="append",
        default=[],
        help="Exclude offers whose title contains this substring (case-insensitive). Can be used multiple times.",
    )
    args = p.parse_args(argv)

    default_excludes = [
        "Stagiaire Master",
        "Master Internship",
        "Post Doctorant",
        "Post-Doctoral",
    ]
    excludes = list(default_excludes) + list(args.exclude_title or [])

    crawl(
        start_url=args.start_url,
        db_path=args.db,
        with_details=args.with_details,
        max_pages=args.max_pages,
        delay_s=max(0.0, args.delay),
        timeout_s=max(1.0, args.timeout),
        retries=max(0, args.retries),
        backoff_base_s=max(0.1, args.backoff),
        user_agent=args.user_agent,
        excluded_title_tokens=excludes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
