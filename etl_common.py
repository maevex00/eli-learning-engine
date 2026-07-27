"""
Shared ETL logic: MailChimp/Brevo + Backend -> eli-intel DB.

Sources (matches eliworks-mailchimp-dashboard/app.py's methodology):
  MailChimp/Brevo -> open_rate, unique_opens, emails_sent (net of bounces), axis_primary,
               conversation_id, cta_text (all parsed from campaign HTML content —
               conversation_id from the cv= param, cta_text from the EMOTE PROMPT/CTA
               section, both in one fetch_campaign_extras()/fetch_campaign_detail_brevo()
               request per campaign)
  Backend   -> emoji_clicks         : read from public.eli_intel_channel (see
                                       channel_cleanser.py), which this ETL rebuilds at
                                       the start of every run from {schema}.chat_votepayload
                                       WHERE vote_status = 'valid', grouped by
                                       (conversation_id, channel). vote_status itself is
                                       still written by the production Vote Cleanser —
                                       this only re-aggregates its output with a channel
                                       dimension that eli_intel/eli_intelemoji's own
                                       rebuild doesn't have (see channel_cleanser.py's
                                       docstring for why that matters).
               conversation_starts  : COUNT(DISTINCT "user") FROM {schema}.chat_userreply
                                       WHERE created_at >= campaign send_date
                                       (same query as app.py fetch_conversation_counts)
  eli_intel -> landing_page_opens   : per-channel counts from the public.eli_intel table's
                                       landing_page_opens_by_channel jsonb column (keyed by
                                       conversation_id), which the Vote Cleanser keeps fresh
                                       every 15 minutes — this is the intended source per
                                       Maeve's boss (2026-07-10), replacing the old
                                       landing_page_opens_text/_qr columns that nothing wrote to.

Match: campaign -> subject_line_library by subject_line + send_date, regardless of
sending platform. A campaign with no matching row is a new subject line — it gets
INSERTed into subject_line_library (+ a paired engagement_metrics row) rather than
skipped, so newly-sent campaigns are picked up automatically on the next ETL run.

Per-candidate entry scripts:
  MailChimp — etl_mailchimp_wiley.py, etl_mailchimp_joy_eakins.py, etl_mailchimp_czajka.py
              set CANDIDATE / SCHEMA / MC key+dc and call run().
  Brevo     — etl_brevo_wiley.py, etl_brevo_go_america_pac.py
              set CANDIDATE / SCHEMA / campaign prefix and call run_brevo().
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path

import psycopg2
import requests

import channel_cleanser

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


# -- Brevo helpers ----------------------------------------------------------------
# Raw HTTP against Brevo's Marketing API v3, same style as mc_get() above — this
# deliberately does NOT import Launcher's adapters/brevo.py (a separate repo not
# checked out in this repo's GitHub Actions run). That adapter's list_campaigns()/
# fetch_analytics() were used as the reference for endpoint shapes and auth only.
BREVO_BASE = "https://api.brevo.com/v3"


def brevo_get(brevo_key, path, params=None):
    r = requests.get(
        f"{BREVO_BASE}{path}",
        headers={"api-key": brevo_key, "Accept": "application/json"},
        params=params or {},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def fetch_all_sent_campaigns_brevo(brevo_key, campaign_prefix):
    """All sent classic campaigns whose name starts with campaign_prefix. Client-side
    prefix filter (Brevo's list endpoint has no name-prefix param) — matches
    adapters/brevo.py's list_campaigns() approach. One Brevo account/key is shared
    across clients, so the prefix is what scopes results to one candidate."""
    results = []
    offset, limit = 0, 50
    while True:
        data = brevo_get(brevo_key, "/emailCampaigns", {
            "limit": limit, "offset": offset, "type": "classic", "status": "sent",
        })
        campaigns = data.get("campaigns") or []
        for c in campaigns:
            if (c.get("name") or "").startswith(campaign_prefix):
                results.append(c)
        total = int(data.get("count", 0))
        offset += limit
        if offset >= total or not campaigns:
            break
    return results


def fetch_campaign_detail_brevo(brevo_key, campaign_id):
    """Full campaign detail (globalStats + htmlContent + subject + sentDate) for one
    campaign. NOTE: field names (subject/htmlContent/sentDate) are per Brevo's
    documented emailCampaigns schema but unverified against this project's live
    account — confirm on the first real run before this goes into the nightly matrix."""
    return brevo_get(brevo_key, f"/emailCampaigns/{campaign_id}", {"statistics": "globalStats"})


# -- Backend (Postgres) helpers ---------------------------------------------------
def fetch_emoji_clicks(conn, client_id, conv_ids):
    """Emoji click counts per conversation, email-channel only — reads the channel-aware
    clean aggregate in public.eli_intel_channel (see channel_cleanser.py), which run()/
    run_brevo() rebuild at the start of every ETL run from this client's own
    chat_votepayload.vote_status. Only 'valid'-tagged rows are counted — untagged (NULL,
    not yet processed by the Vote Cleanser) rows are excluded rather than assumed valid.

    Scoping to channel='email' here (on top of eli_intel_channel already being grouped
    by channel) matters whenever a conversation_id is shared across channels — without
    it, an email row with 0 opens could inherit another channel's clicks and show an
    impossible >100% click rate (found in production 2026-07-14: conv 969 showed 18
    combined clicks — 7 email + 1 event + 10 unknown — all credited to the email row).
    Backend-only channels are unaffected: sync_backend_channels() scopes its own
    eli_intel_channel query by channel too."""
    if not conv_ids or client_id is None:
        return {}
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, emoji_clicks
        FROM eli_intel_channel
        WHERE client_id = %s AND conversation_id = ANY(%s) AND channel = 'email'
    """, (client_id, list(conv_ids)))
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
# 'email' is handled separately via the MailChimp/Brevo match in build_updates()/
# build_updates_brevo()/write_updates().
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
    match against, so chat_votepayload/chat_userreply (and, for emoji clicks specifically,
    the eli_intel_channel rebuild of chat_votepayload — see fetch_emoji_clicks()) are the
    ONLY source of truth. subject_line_library/engagement_metrics are write targets here,
    never a source.
    """
    channels = discover_backend_channels(conn, schema)
    if not channels:
        print(f"  No backend-only channels found for {candidate}.")
        return 0

    cur = conn.cursor()

    cur.execute("SELECT id, name FROM eli_conversation WHERE client_id = %s", (client_id,))
    conv_names = {r[0]: r[1] for r in cur.fetchall()}

    # Same eli_intel_channel table fetch_emoji_clicks() reads for email — rebuilt at the
    # top of run()/run_brevo(), so it's already fresh for this call.
    cur.execute("""
        SELECT conversation_id, channel, emoji_clicks
        FROM eli_intel_channel
        WHERE client_id = %s AND channel = ANY(%s)
    """, (client_id, channels))
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

    enrich_with_backend_data(conn, schema, client_id, updates)
    return updates


def enrich_with_backend_data(conn, schema, client_id, updates):
    """Fill in emoji_clicks/emoji_click_rate, conversation_starts/conversation_rate,
    and landing_page_opens on each dict in `updates`, in place. Platform-agnostic —
    chat_votepayload/chat_userreply/eli_intel are the same backend tables regardless
    of whether the campaign data came from MailChimp or Brevo, so this is shared by
    build_updates() and build_updates_brevo()."""
    conv_ids = {u["conversation_id"] for u in updates if u["conversation_id"]}

    # -- Backend emoji clicks (eli_intel_channel, rebuilt from chat_votepayload) -----
    print(f"\nFetching emoji clicks from eli_intel_channel for {len(conv_ids)} conversation(s)...")
    emoji_counts = fetch_emoji_clicks(conn, client_id, conv_ids)

    for u in updates:
        cid = u["conversation_id"]
        # A known conv_id absent from emoji_counts means zero valid votes were found
        # for it (GROUP BY omits zero-count groups) — that's 0, not unknown. But if
        # client_id itself is missing we can't query eli_intel_channel at all, so the
        # count is genuinely unknown rather than a confirmed zero.
        emoji_clicks = emoji_counts.get(cid, 0) if (cid and client_id is not None) else None
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


def build_updates_brevo(conn, candidate, schema, brevo_key, campaign_prefix, client_id=None):
    """Brevo counterpart to build_updates(). Same dedup key (subject_line + send_date
    against subject_line_library, via load_candidate_library()) and same backend
    enrichment (enrich_with_backend_data()) — only the campaign-list/report source
    differs."""
    lib        = load_candidate_library(conn, candidate)
    b_campaigns = fetch_all_sent_campaigns_brevo(brevo_key, campaign_prefix)
    print(f"DB: {len(lib)} {candidate} email rows")
    print(f"Brevo: {len(b_campaigns)} sent campaigns matching '{campaign_prefix}'\n")

    updates = []
    for c in b_campaigns:
        campaign_id = c["id"]
        title       = (c.get("name") or "").strip()

        try:
            detail = fetch_campaign_detail_brevo(brevo_key, campaign_id)
            time.sleep(0.2)
        except Exception as e:
            print(f"  WARN detail {campaign_id}: {e}")
            continue

        subject = (detail.get("subject") or "").strip()
        # sentDate is the actual send timestamp. scheduledAt/createdAt are NOT a
        # reliable fallback for immediately-sent campaigns (createdAt is draft-creation
        # time, which can predate the real send by days) — a wrong send_date here
        # breaks the subject_line+send_date dedup key and creates duplicate rows.
        send_time = detail.get("sentDate") or detail.get("scheduledAt") or detail.get("createdAt", "")
        if not subject or not send_time:
            print(f"  SKIP {campaign_id}: missing subject or send date")
            continue

        send_date = send_time[:10]
        db_row    = lib.get((subject.lower(), send_date))
        is_new    = db_row is None

        stats = (detail.get("statistics") or {}).get("globalStats") or {}
        delivered    = max(0, int(stats.get("delivered", 0)))
        # Verified against a live campaign 2026-07-24: globalStats has no "uniqueOpens"
        # key at all (that name is also wrong in Launcher's adapters/brevo.py, whose
        # fetch_analytics() has been silently returning 0 unique opens for Brevo
        # clients — flag this back to whoever owns the Launcher repo). The real field
        # is "uniqueViews"; "viewed" is total (non-unique) opens.
        unique_opens = int(stats.get("uniqueViews", 0))
        # Whether uniqueViews already excludes Apple MPP machine-prefetch opens is
        # unconfirmed — globalStats also has a separate "appleMppOpens" counter that
        # doesn't obviously net out of uniqueViews in the one campaign checked so far.
        # Treat as directional only, not directly comparable to MailChimp clients'
        # proxy_excluded numbers. Stored as a 0-1 fraction to match the MailChimp ETL's
        # column convention. The dashboard recomputes its displayed open rate from raw
        # counts anyway, so this only matters if something else reads the column directly.
        open_rate = round(unique_opens / delivered, 6) if delivered else None

        html = detail.get("htmlContent", "") or ""
        m = re.search(r"[?&]cv=(\d+)", html, re.IGNORECASE)
        conv_id  = int(m.group(1)) if m else None
        cta_text = extract_cta(html)

        axis = extract_axis(title)

        updates.append({
            "sl_id":            None if is_new else db_row["id"],
            "is_new":           is_new,
            "subject_line":     subject,
            "campaign":         title,
            "mc_campaign_id":   campaign_id,  # column is platform-agnostic despite the name
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

    enrich_with_backend_data(conn, schema, client_id, updates)
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

    if client_id is not None and not dry_run:
        channel_cleanser.rebuild_channel_emoji_clicks(conn, schema, client_id)

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


def run_brevo(candidate, schema, brevo_key, campaign_prefix, client_id=None, dry_run=False):
    """Brevo counterpart to run(). campaign_prefix scopes the shared Brevo account's
    campaign list to this candidate, e.g. "eli-james_wiley-"."""
    print(f"[{datetime.now()}] Starting Brevo ETL for {candidate}{' [DRY RUN]' if dry_run else ''}\n")

    conn = psycopg2.connect(**DB_CONN)

    if client_id is not None and not dry_run:
        channel_cleanser.rebuild_channel_emoji_clicks(conn, schema, client_id)

    updates = build_updates_brevo(conn, candidate, schema, brevo_key, campaign_prefix, client_id=client_id)

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
    print(f"\n[{datetime.now()}] Brevo ETL complete for {candidate}.\n")
    return updates
