"""
Shared ETL logic: MailChimp + Backend -> eli-intel DB.

Sources (matches eliworks-mailchimp-dashboard/app.py's methodology):
  MailChimp -> open_rate, unique_opens, emails_sent (net of bounces), axis_primary,
               conversation_id, cta_text (all parsed from campaign HTML content —
               conversation_id from the cv= param, cta_text from the EMOTE PROMPT/CTA
               section, both in one fetch_campaign_extras() request per campaign)
  Backend   -> emoji_clicks         : COUNT(*) FROM {schema}.chat_votepayload
                                       WHERE vote_status = 'valid' OR vote_status IS NULL
                                       (same query as app.py fetch_vote_counts — bot-tagged
                                       votes from the nightly Vote Cleanser are excluded)
               conversation_starts  : COUNT(DISTINCT "user") FROM {schema}.chat_userreply
                                       WHERE created_at >= campaign send_date
                                       (same query as app.py fetch_conversation_counts)
  eli_intel -> landing_page_opens   : per-channel counts from the public.eli_intel table's
                                       landing_page_opens_by_channel jsonb column (keyed by
                                       conversation_id), which the Vote Cleanser keeps fresh
                                       every 15 minutes — this is the intended source per
                                       Maeve's boss (2026-07-10), replacing the old
                                       landing_page_opens_text/_qr columns that nothing wrote to.

Match: MailChimp campaign -> subject_line_library by subject_line + send_date.
A MailChimp campaign with no matching row is a new subject line — it gets INSERTed
into subject_line_library (+ a paired engagement_metrics row) rather than skipped,
so newly-sent campaigns are picked up automatically on the next ETL run.

Per-candidate entry scripts (etl_mailchimp_wiley.py, etl_mailchimp_joy_eakins.py,
etl_mailchimp_czajka.py) just set CANDIDATE / SCHEMA / MC key+dc and call run().
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path

import psycopg2
import requests

# Load a local .env (gitignored) for local runs — GitHub Actions sets real env
# vars via repo Secrets instead, so this is a no-op there.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DB_CONN = dict(
    host=os.environ["DB_HOST"],
    port=int(os.environ.get("DB_PORT", 5432)),
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
)


# -- MailChimp helpers ----------------------------------------------------------
def mc_get(mc_base, mc_auth, path, params=None):
    r = requests.get(f"{mc_base}/{path}", auth=mc_auth, params=params or {})
    r.raise_for_status()
    return r.json()


def fetch_all_sent_campaigns(mc_base, mc_auth):
    data = mc_get(mc_base, mc_auth, "campaigns", {
        "count": 1000,
        "status": "sent",
        "fields": "campaigns.id,campaigns.settings.title,campaigns.settings.subject_line,"
                  "campaigns.send_time,campaigns.emails_sent",
    })
    return data.get("campaigns", [])


def extract_axis(campaign_title):
    """
    Extract axis from Eli campaign title.
    Pattern: eli-{candidate}-{Axis}-SL{num}-cv{num}
    Axis may be one or more words (e.g. "Issues", "Election Integrity") but never
    contains a hyphen itself, since "-SL{num}" is what terminates it.
    Returns Title-cased axis string, or None if not an Eli campaign.
    """
    if not campaign_title:
        return None
    m = re.match(r'^eli-[^-]+-([A-Za-z][A-Za-z ]*?)-SL\d+', campaign_title, re.IGNORECASE)
    if m:
        return m.group(1).title()   # normalize: issues->Issues, election integrity->Election Integrity
    return None


def _clean_html_text(s):
    """Strip tags/entities down to plain text — same cleanup as app.py's
    parse_greeting_cta()'s inner clean()."""
    s = re.sub(r'<br\s*/?>', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'&nbsp;', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'&amp;', '&', s, flags=re.IGNORECASE)
    s = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), s)
    s = re.sub(r'&[a-zA-Z#0-9]+;', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def extract_cta(html):
    """CTA text from the campaign HTML — checks every section-comment variant seen in
    production templates: '<!-- EMOTE PROMPT -->', '<!-- CTA -->', and (confirmed on
    James Wiley's live template 2026-07-10) '<!-- Question / CTA -->', which sits right
    above the emoji grid and is where the real prompt text actually lives — the
    'EMOTE PROMPT' section on that same template is present but structurally empty.
    Matches any comment containing the word CTA, not just an exact '<!-- CTA -->'."""
    if not html:
        return None
    for section_pattern in (r'EMOTE PROMPT', r'[^>]*\bCTA\b[^>]*'):
        m = re.search(
            rf'<!--\s*{section_pattern}\s*-->(.*?)(?=<!--|\Z)',
            html, re.DOTALL | re.IGNORECASE
        )
        if not m:
            continue
        for p in re.finditer(r'<p[^>]*>(.*?)</p>', m.group(1), re.DOTALL | re.IGNORECASE):
            txt = _clean_html_text(p.group(1))
            if len(txt) > 10:
                return txt[:300]
    return None


def fetch_campaign_extras(mc_base, mc_auth, campaign_id):
    """Conversation id (cv= param) and CTA text, both parsed from a single fetch of
    the campaign's HTML content — same approach as app.py's get_campaign_content()."""
    data = mc_get(mc_base, mc_auth, f"campaigns/{campaign_id}/content")
    html = data.get("html", "") or data.get("html_clean", "")
    m = re.search(r"[?&]cv=(\d+)", html, re.IGNORECASE)
    conv_id = int(m.group(1)) if m else None
    return conv_id, extract_cta(html)


# -- Backend (Postgres) helpers ---------------------------------------------------
def fetch_emoji_clicks(conn, schema, conv_ids):
    """Emoji click counts per conversation, email-channel only, from chat_votepayload —
    identical query to the reference dashboard's fetch_vote_counts(), plus a channel
    filter. Rows tagged vote_status='bot' by the nightly Vote Cleanser are excluded;
    untagged (NULL) rows count as valid.

    The channel filter matters whenever a conversation_id is shared across channels
    (e.g. a QR-code conversation that also picked up email/event/unknown-channel votes) —
    without it, an email row with 0 opens could inherit another channel's clicks and
    show an impossible >100% click rate (found in production 2026-07-14: conv 969 showed
    18 combined clicks — 7 email + 1 event + 10 unknown — all credited to the email row,
    which had 0 opens). Backend-only channels are unaffected: sync_backend_channels()
    already scopes its own query by channel."""
    if not conv_ids:
        return {}
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, COUNT(*)
        FROM "{schema}".chat_votepayload
        WHERE conversation_id = ANY(%s)
          AND channel = 'email'
          AND (vote_status = 'valid' OR vote_status IS NULL)
        GROUP BY conversation_id
    """.format(schema=schema), (list(conv_ids),))
    result = {r[0]: r[1] for r in cur.fetchall()}
    cur.close()
    return result


def fetch_conversation_starts(conn, schema, conv_send_dates):
    """Distinct repliers per conversation on/after the campaign's send date, email-channel
    only, from chat_userreply — identical query to the reference dashboard's
    fetch_conversation_counts(), plus a channel filter (same rationale as
    fetch_emoji_clicks() above — a shared conversation_id shouldn't let another
    channel's replies inflate the email row). conv_send_dates: {conv_id: 'YYYY-MM-DD'}."""
    result = {}
    cur = conn.cursor()
    for conv_id, send_date in conv_send_dates.items():
        cur.execute("""
            SELECT COUNT(DISTINCT "user")
            FROM "{schema}".chat_userreply
            WHERE conversation_id = %s AND channel = 'email' AND created_at >= %s
        """.format(schema=schema), (conv_id, send_date))
        row = cur.fetchone()
        result[conv_id] = row[0] if row else 0
    cur.close()
    return result


def fetch_landing_page_opens_by_channel(conn, client_id, conv_ids):
    """{conversation_id: {channel: opens}} from public.eli_intel.landing_page_opens_by_channel
    for this client's conversations — eli_intel is keyed one row per conversation_id (verified
    no duplicates), refreshed by the same Vote Cleanser that tags chat_votepayload."""
    if not conv_ids:
        return {}
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, landing_page_opens_by_channel
        FROM eli_intel
        WHERE client_id = %s AND conversation_id = ANY(%s)
    """, (client_id, list(conv_ids)))
    result = {r[0]: (r[1] or {}) for r in cur.fetchall()}
    cur.close()
    return result


# -- Backend-only channels (no MailChimp campaign to match against) ------------------
# 'email' is handled separately via the MailChimp match in build_updates()/write_updates().
# Everything else is discovered dynamically from chat_votepayload/chat_userreply's own
# `channel` column, so a brand-new channel (Website, Direct Mail, ...) starts showing up
# on the dashboard automatically the first time it produces real rows — no code change
# needed. Denylist below filters out placeholder/junk values confirmed NOT to be real
# marketing channels (checked against production data 2026-07-10): blank/NULL, 'unknown',
# 'native' and 'harness' (single-digit noise on a retired client), and 'widget' (an internal
# EliWorks test conversation, not a candidate-facing channel). 'text'/'qr'/'event'/'social'
# were confirmed real and no longer need special-casing — they just fall out of discovery.
EXCLUDED_CHANNEL_VALUES = {None, '', 'email', 'unknown', 'native', 'harness', 'widget'}


def discover_backend_channels(conn, schema):
    """Distinct non-email channel values actually present in this schema's backend
    tables right now, minus the known-junk denylist above."""
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT channel FROM "{schema}".chat_votepayload'.format(schema=schema))
    values = {r[0] for r in cur.fetchall()}
    cur.execute('SELECT DISTINCT channel FROM "{schema}".chat_userreply'.format(schema=schema))
    values |= {r[0] for r in cur.fetchall()}
    cur.close()
    return sorted(v for v in values if v not in EXCLUDED_CHANNEL_VALUES)


def sync_backend_channels(conn, candidate, client_id, schema):
    """Backend-only channels (see discover_backend_channels) have no MailChimp campaign to
    match against, so chat_votepayload/chat_userreply's own `channel` column is the
    ONLY source of truth. subject_line_library/engagement_metrics are write
    targets here, never a source.
    """
    channels = discover_backend_channels(conn, schema)
    if not channels:
        print(f"  No backend-only channels found for {candidate}.")
        return 0

    cur = conn.cursor()

    cur.execute("SELECT id, name FROM eli_conversation WHERE client_id = %s", (client_id,))
    conv_names = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("""
        SELECT conversation_id, channel, COUNT(*)
        FROM "{schema}".chat_votepayload
        WHERE channel = ANY(%s)
          AND (vote_status = 'valid' OR vote_status IS NULL)
        GROUP BY conversation_id, channel
    """.format(schema=schema), (channels,))
    emoji_by_key = {(r[0], r[1]): r[2] for r in cur.fetchall()}

    cur.execute("""
        SELECT conversation_id, channel, COUNT(DISTINCT "user")
        FROM "{schema}".chat_userreply
        WHERE channel = ANY(%s)
        GROUP BY conversation_id, channel
    """.format(schema=schema), (channels,))
    starts_by_key = {(r[0], r[1]): r[2] for r in cur.fetchall()}

    keys = sorted(set(emoji_by_key) | set(starts_by_key))
    print(f"  Backend channels found: {channels} — {len(keys)} (conversation, channel) pair(s)")

    lp_by_conv = fetch_landing_page_opens_by_channel(conn, client_id, {k[0] for k in keys})

    written = 0
    for conv_id, channel in keys:
        emoji_clicks = emoji_by_key.get((conv_id, channel), 0)
        starts       = starts_by_key.get((conv_id, channel), 0)
        rate         = round(starts / emoji_clicks, 6) if emoji_clicks else None
        label        = conv_names.get(conv_id) or f"Conversation {conv_id}"
        lp_opens     = lp_by_conv.get(conv_id, {}).get(channel)

        cur.execute("""
            SELECT id FROM subject_line_library
            WHERE candidate = %s AND channel = %s AND conversation_id = %s
        """, (candidate, channel, conv_id))
        row = cur.fetchone()
        if row:
            sl_id = row[0]
        else:
            cur.execute("""
                INSERT INTO subject_line_library
                    (candidate, campaign, subject_line, channel, conversation_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (candidate, label, label, channel, conv_id))
            sl_id = cur.fetchone()[0]

        cur.execute("""
            UPDATE engagement_metrics
            SET emoji_clicks        = %s,
                conversation_starts = %s,
                conversation_rate   = %s,
                landing_page_opens  = %s
            WHERE subject_line_id = %s AND channel = %s
        """, (emoji_clicks, starts, rate, lp_opens, sl_id, channel))
        if cur.rowcount == 0:
            cur.execute("""
                INSERT INTO engagement_metrics
                    (subject_line_id, channel, emoji_clicks, conversation_starts, conversation_rate,
                     landing_page_opens)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (sl_id, channel, emoji_clicks, starts, rate, lp_opens))
        written += 1

    conn.commit()
    cur.close()
    print(f"  {written} backend-channel row(s) written for {candidate}.")
    return written


# -- DB helpers -------------------------------------------------------------------
def load_candidate_library(conn, candidate):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, subject_line, send_date, emails_sent, campaign
        FROM subject_line_library
        WHERE candidate = %s AND channel = 'email'
    """, (candidate,))
    rows = {}
    for id_, subject, send_date, emails_sent, campaign in cur.fetchall():
        key = (subject.strip().lower(), str(send_date))
        rows[key] = {
            "id": id_,
            "subject_line": subject,
            "send_date": str(send_date),
            "emails_sent": emails_sent,
            "campaign": campaign,
        }
    cur.close()
    return rows


# -- Main ETL -----------------------------------------------------------------------
def build_updates(conn, candidate, schema, mc_base, mc_auth, client_id=None):
    """Fetch MailChimp + backend data and return the list of per-campaign update
    dicts, without writing anything to the DB. Shared by run() and dry-run tooling."""
    lib          = load_candidate_library(conn, candidate)
    mc_campaigns = fetch_all_sent_campaigns(mc_base, mc_auth)
    print(f"DB: {len(lib)} {candidate} email rows")
    print(f"MailChimp: {len(mc_campaigns)} sent campaigns\n")

    updates = []
    for mc in mc_campaigns:
        subject   = mc["settings"].get("subject_line", "").strip()
        send_time = mc.get("send_time", "")
        if not subject or not send_time:
            continue

        send_date = send_time[:10]
        db_row    = lib.get((subject.lower(), send_date))
        is_new    = db_row is None

        try:
            report = mc_get(mc_base, mc_auth, f"reports/{mc['id']}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  WARN report {mc['id']}: {e}")
            continue

        # "Sent" = delivered (net of bounces), matching app.py's delivered calc —
        # not the raw emails_sent MailChimp API field.
        emails_sent_raw = report.get("emails_sent", 0)
        bounces         = report.get("bounces") or {}
        delivered       = max(0, emails_sent_raw - bounces.get("hard_bounces", 0)
                                                   - bounces.get("soft_bounces", 0))

        opens        = report.get("opens", {})
        open_rate    = opens.get("proxy_excluded_open_rate")  # excludes Apple MPP fake opens
        unique_opens = opens.get("proxy_excluded_unique_opens")

        try:
            conv_id, cta_text = fetch_campaign_extras(mc_base, mc_auth, mc["id"])
            time.sleep(0.2)
        except Exception as e:
            print(f"  WARN content fetch {mc['id']}: {e}")
            conv_id, cta_text = None, None

        title = mc["settings"].get("title", "").strip()
        axis  = extract_axis(title)

        updates.append({
            "sl_id":            None if is_new else db_row["id"],
            "is_new":           is_new,
            "subject_line":     subject,
            "campaign":         title,
            "mc_campaign_id":   mc["id"],
            "send_date":        send_date,
            "emails_sent":      delivered,
            "open_rate":        open_rate,
            "unique_opens":     unique_opens,
            "conversation_id":  conv_id,
            "axis_primary":     axis,
            "cta_text":         cta_text,
        })
        tag = "NEW" if is_new else f"[{db_row['id']}]"
        print(f"  OK {tag} {send_date} | {subject[:50]}")
        print(f"       delivered={delivered}  unique={unique_opens}  conv_id={conv_id}  axis={axis}")

    n_new = sum(1 for u in updates if u["is_new"])
    print(f"\nMatched {len(updates)} campaigns ({n_new} new, {len(updates) - n_new} existing).")

    # -- Backend emoji clicks (chat_votepayload, bot-filtered) --------------------
    conv_ids = {u["conversation_id"] for u in updates if u["conversation_id"]}
    print(f"\nFetching emoji clicks from {schema}.chat_votepayload for {len(conv_ids)} conversation(s)...")
    emoji_counts = fetch_emoji_clicks(conn, schema, conv_ids)

    for u in updates:
        cid = u["conversation_id"]
        # A known conv_id absent from emoji_counts means zero valid votes were
        # found for it (GROUP BY omits zero-count groups) — that's 0, not unknown.
        emoji_clicks = emoji_counts.get(cid, 0) if cid else None
        u["emoji_clicks"] = emoji_clicks
        # Funnel rate: each stage divides by the PREVIOUS stage, not by emails_sent.
        u["emoji_click_rate"] = (
            round(emoji_clicks / u["unique_opens"], 6)
            if emoji_clicks is not None and u["unique_opens"]
            else None
        )

    # -- Backend conversation starts (chat_userreply) ------------------------------
    conv_send_dates = {u["conversation_id"]: u["send_date"] for u in updates if u["conversation_id"]}
    print(f"Fetching conversation starts from {schema}.chat_userreply for {len(conv_send_dates)} conversation(s)...")
    starts_by_conv = fetch_conversation_starts(conn, schema, conv_send_dates)

    for u in updates:
        cid    = u["conversation_id"]
        starts = starts_by_conv.get(cid, 0) if cid else None
        u["conversation_starts"] = starts
        # Rate is starts / emoji_clicks (previous funnel stage), not starts / emails_sent.
        u["conversation_rate"] = (
            round(starts / u["emoji_clicks"], 6)
            if starts and u.get("emoji_clicks")
            else None
        )

    # -- Landing page opens, per channel (eli_intel) --------------------------------
    if client_id is not None:
        lp_by_conv = fetch_landing_page_opens_by_channel(conn, client_id, conv_ids)
        for u in updates:
            cid = u["conversation_id"]
            u["landing_page_opens"] = lp_by_conv.get(cid, {}).get("email") if cid else None
    else:
        for u in updates:
            u["landing_page_opens"] = None

    return updates


def write_updates(conn, candidate, updates):
    cur = conn.cursor()

    # Clear existing conversation_starts/rate for this candidate's email rows so
    # campaigns that no longer resolve to a conv_id (or have zero backend replies)
    # don't keep a stale value from a previous ETL run.
    cur.execute("""
        UPDATE engagement_metrics em
        SET conversation_starts = NULL,
            conversation_rate   = NULL
        FROM subject_line_library sl
        WHERE em.subject_line_id = sl.id
          AND sl.candidate = %s
          AND em.channel = 'email'
    """, (candidate,))
    print(f"  Cleared {cur.rowcount} conversation_starts row(s)")
    conn.commit()

    em_updated = 0
    new_count  = 0
    for u in updates:
        if u["sl_id"] is None:
            # New subject line MailChimp knows about that isn't in the corpus yet —
            # insert it rather than silently dropping it.
            cur.execute("""
                INSERT INTO subject_line_library
                    (candidate, campaign, subject_line, channel, send_date,
                     mailchimp_campaign_id, emails_sent, conversation_id, axis_primary, cta_text)
                VALUES (%s, %s, %s, 'email', %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (candidate, u["campaign"], u["subject_line"], u["send_date"],
                  u["mc_campaign_id"], u["emails_sent"], u["conversation_id"], u["axis_primary"],
                  u["cta_text"]))
            u["sl_id"] = cur.fetchone()[0]
            new_count += 1
        else:
            cur.execute("""
                UPDATE subject_line_library
                SET
                    emails_sent           = %s,
                    mailchimp_campaign_id = %s,
                    conversation_id       = %s,
                    axis_primary          = COALESCE(axis_primary, %s),
                    cta_text               = %s
                WHERE id = %s
            """, (u["emails_sent"], u["mc_campaign_id"], u["conversation_id"],
                  u["axis_primary"], u["cta_text"], u["sl_id"]))

        cur.execute("""
            UPDATE engagement_metrics
            SET
                open_rate            = %s,
                unique_opens         = %s,
                emoji_clicks         = %s,
                emoji_click_rate     = %s,
                conversation_starts  = %s,
                conversation_rate    = %s,
                landing_page_opens   = %s
            WHERE subject_line_id = %s AND channel = 'email'
        """, (
            u["open_rate"], u["unique_opens"], u["emoji_clicks"], u["emoji_click_rate"],
            u["conversation_starts"], u["conversation_rate"], u["landing_page_opens"],
            u["sl_id"],
        ))
        if cur.rowcount == 0:
            # Brand-new subject_line_library row (or one that never got an
            # engagement_metrics counterpart) — insert instead of update.
            cur.execute("""
                INSERT INTO engagement_metrics
                    (subject_line_id, channel, open_rate, unique_opens, emoji_clicks,
                     emoji_click_rate, conversation_starts, conversation_rate, landing_page_opens)
                VALUES (%s, 'email', %s, %s, %s, %s, %s, %s, %s)
            """, (
                u["sl_id"], u["open_rate"], u["unique_opens"], u["emoji_clicks"],
                u["emoji_click_rate"], u["conversation_starts"], u["conversation_rate"],
                u["landing_page_opens"],
            ))
        em_updated += 1

    conn.commit()
    cur.close()
    print(f"  {em_updated} engagement_metrics row(s) written ({new_count} new subject line(s) added).")


def run(candidate, schema, mc_key, mc_dc, client_id=None, dry_run=False):
    mc_base = f"https://{mc_dc}.api.mailchimp.com/3.0"
    mc_auth = ("anystring", mc_key)

    print(f"[{datetime.now()}] Starting ETL for {candidate}{' [DRY RUN]' if dry_run else ''}\n")

    conn = psycopg2.connect(**DB_CONN)
    updates = build_updates(conn, candidate, schema, mc_base, mc_auth, client_id=client_id)

    if dry_run:
        conn.close()
        print(f"\n[DRY RUN] No DB writes performed for {candidate}.")
        return updates

    print("Writing to DB...")
    write_updates(conn, candidate, updates)

    if client_id is not None:
        print(f"\nSyncing non-email backend channels for {candidate}...")
        sync_backend_channels(conn, candidate, client_id, schema)

    conn.close()
    print(f"\n[{datetime.now()}] ETL complete for {candidate}.\n")
    return updates
